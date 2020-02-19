# @author: niwics, niwi.cz, September 2018

import logging
import argparse
import datetime
import dateutil.parser
import re
from copy import deepcopy
import os
import sys

from gpxpy import gpx
from gpxpy import geo
from gpxpy import parse as gpx_parse
from xml.etree import ElementTree

GARMIN_NS = 'http://www.garmin.com/xmlschemas/TrackPointExtension/v1'
HR_BASE_TAG = "{{{}}}TrackPointExtension".format(GARMIN_NS)
HR_TAG = "{{{}}}hr".format(GARMIN_NS)
PACE_RE_PATTERN = r'(\d\d?):(\d\d)'
FIX_TIME_PATTERN = r'((\d?\d):)?(\d?\d):(\d\d)'
# shortest distance after which the pause will be generated (when filling times)
PAUSE_LIMIT_METERS = 200
# max allowed distance (in meters) between two points
MAX_POINT_DISTANCE = 20

# logging setup
log = logging.getLogger(__name__)
ch = logging.StreamHandler()
log.addHandler(ch)

class StravaGpxException(Exception):
    pass

class PointsBoundaries:
    # TODO start/end attrs could have int or datetime values now => separate
    def __init__(self, start, end):
        if not start and not end:
            raise ValueError('One of start or end values must be set')
        if start > end and start and end:   # just in case both are set
            raise ValueError('End value must be higher than start value')
        self.start = start
        self.end = end
    
    @property
    def start(self):
        return self.__start
    
    @property
    def end(self):
        return self.__end
    
    def __str__(self):
        return '({},{})'.format(self.start, self.end)

class StravaGpxTool:
    """Main processing class."""

    @staticmethod
    def parse_pace(pace):
        pace_re = re.search(PACE_RE_PATTERN, pace)
        if not pace_re:
            raise StravaGpxException("Invalid format of pace: {}. Should be in format MM:SS.".format(pace))
        return pace
    
    @staticmethod
    def compute_path_length(points):
        """Computes the total distance of path points.
        
        :param points: List of GPX tracks
        :type points: [gpxpy.gpx.GPXPoint]
        :returns: Path length in meters
        :rtype: float
        """
        length = 0
        prev_point = None
        for point in points:
            if prev_point:
                length += geo.length_3d((prev_point, point))
            prev_point = point
        return length
    
    @staticmethod
    def get_points_from_file(filename):
        try:
            in_file = open(filename, 'r')
        except IOError as e:
            raise StravaGpxException("Could not open file {}: {}".format(filename, e))
        try:
            in_gpx = gpx_parse(in_file)
        except gpx.GPXXMLSyntaxException as e:
            raise StravaGpxException("Error while parsing file {}: {}".format(filename, e))
        return StravaGpxTool.extract_points(in_gpx.tracks)
    
    @staticmethod
    def extract_points(tracks):
        if len(tracks) == 0:
            raise StravaGpxException('No GPX tracks found in the input file')
        elif len(tracks) > 1:
            raise StravaGpxException('Multiple GPX tracks in the input file are not supported')
        
        if len(tracks[0].segments) == 0:
            raise StravaGpxException('No GPX segments found in the input file')
        elif len(tracks[0].segments) > 1:
            raise StravaGpxException('Multiple GPX segments in the input file are not supported')
        
        return tracks[0].segments[0].points
    
    @staticmethod
    def get_first_point_datetime(points):
        if not points:
            raise StravaGpxException('No path points found while processing')
        return points[0].time
    
    
    @staticmethod
    def get_point_at_distance(points, distance_to_find):
        assert points, 'Points must be non-empty'
        total_distance = 0
        prev_point = None
        for point in points:
            total_distance += geo.length_3d((prev_point, point))
            if total_distance >= distance_to_find:
                return point
            prev_point = point
        raise StravaGpxException('The input distance {} is higher than path length {}!'.format(
            distance_to_find, total_distance
        ))

    
    @staticmethod
    def get_point_at_time(points, time_to_find):
        for point in points:
            if time_to_find <= point.time:
                return point
        raise StravaGpxException('Point at given time {} was not found in the path.'.format(time_to_find))
    
    @staticmethod
    def get_time_at_index(points, index):
        return points[index].time
    
    @staticmethod
    def get_closest_point_index(points, point_to_find, search_from_end=False):
        assert points, 'Points must be non-empty'
        best_index = 0
        best_distance = sys.maxsize
        i = 0
        points_to_iterate = reversed(points) if search_from_end else points
        for point in points_to_iterate:
            distance_to = geo.length_3d((point_to_find, point))
            if distance_to < best_distance:
                best_distance = distance_to
                best_index = i
            i += 1
        return len(points)-1-best_index if search_from_end else best_index

    def __init__(self, opts):
        """Class constructor.

        :param opts: Dict of program options (arguments)
        :type opts: dict
        """
        self._opts = opts
        self._out_segment = None
        self._time_hr_array = []
        self._time_hr_search_offset = 0
    
    def add_point(self, point):
        """Adds the GPX point for the output file.
        
        :param point: Point to add
        :type point: gpxpy.gpx.GPXPoint
        """
        if not self._out_segment:
            raise StravaGpxException("Could not add point. No output segment defined.")
        self._out_segment.points.append(point)
    
    def get_last_point(self):
        return self._out_segment.points[-1] if self._out_segment.points else None
    
    def dense_points(self, original_points, prev_point_param, next_point):
        points = []
        prev_point = prev_point_param

        def add_points_on_path(from_point, to_point):
            length_from_prev = 0
            if from_point:
                length_from_prev = geo.length_3d((from_point, to_point))
            log.debug('Processing point: {}, length from previous: {}'.format(to_point, length_from_prev))
            if length_from_prev > MAX_POINT_DISTANCE:
                num_points_to_add = int(length_from_prev/MAX_POINT_DISTANCE)
                lat_step_distance = (to_point.latitude - from_point.latitude)/(num_points_to_add+1)
                lon_step_distance = (to_point.longitude - from_point.longitude)/(num_points_to_add+1)
                log.debug('Will add {} points'.format(num_points_to_add))
                for i in range(num_points_to_add):
                    new_point = deepcopy(to_point)
                    new_point.latitude = from_point.latitude + (i+1)*lat_step_distance
                    new_point.longitude = from_point.longitude + (i+1)*lon_step_distance
                    log.debug('New point to add: {}'.format(new_point))
                    points.append(new_point)
    
        for point in original_points:
            add_points_on_path(prev_point, point)
            points.append(point)
            prev_point = point
        # add points towards the next
        if next_point:
            add_points_on_path(prev_point, next_point)

        return points
    
    def add_points(self, points, hr = None, hr_points = None, pace = None,
            fill_time = None, end_next_point=None,
            crop_time = None, crop_index = None,
            next_point = None):

        in_points_counter = out_points_counter = total_distance = 0
        prev_point = self.get_last_point()
        length_from_prev = 0
        current_length_for_pause = 0
        pace_last_time = None

        if crop_index:
            log.debug('Cropping points to indexes: {}'.format(crop_index))
        cropped_points = points[crop_index.start:crop_index.end] if crop_index else points
        if end_next_point:
            end_point = deepcopy(end_next_point)
            end_point.extensions = [] # clear the HR because of its possible filling
            cropped_points.append(end_point)
        dense_points = cropped_points
        if pace:
            dense_points = self.dense_points(cropped_points, prev_point, next_point)

        if pace:
            assert fill_time, 'When setting the pace, both start and end dates must be set'
            assert fill_time.start < fill_time.end, 'End date to set must be higher than start date'
            duration_total = (fill_time.end - fill_time.start).total_seconds()
            pace_re = re.search(PACE_RE_PATTERN, pace)
            assert pace_re, 'Invalid pace format'
            speed_mps = 1000.0 / (int(pace_re.group(1))*60+int(pace_re.group(2)))
            pace_last_time = fill_time.start

        total_length = 0
        elapsed_pause_time = 0
        moving_time = None

        # read and store HR input file
        if hr_points:
            for point in hr_points:
                if point.extensions:
                    for extension_record in point.extensions:
                        if extension_record[0].text:
                            self._time_hr_array.append((point.time, extension_record[0].text))
            print("Stored {} point(s) with HR information from the HR file.".format(len(self._time_hr_array)))

        # compute variables based on overall distance
        if pace:
            total_length = StravaGpxTool.compute_path_length(dense_points)
            moving_time = total_length / speed_mps
            if duration_total < moving_time:
                raise StravaGpxException('Moving time ({}s) computed from the pace ({}) is smaller than time window of fixed activity ({}s). You should set faster pace.'
                    .format(moving_time, pace, duration_total))
            # reserve 10% for the pause at the end (because of little innaccuracies created by un-smoothing the path - see MAX_POINT_DISTANCE)
            pause_time = int((duration_total - moving_time) * 0.9)
            log.info("Filling the pace: moving time: {:.0f} seconds, pause time: {:.0f} seconds, speed: {:.2f} m/s (= {:.2f} km/h = pace {}), distance: {:.0f} m"
                .format(moving_time, pause_time, speed_mps, speed_mps*3.6, pace, total_length))

        log.debug("Started at time: {}".format(pace_last_time))
        for point in dense_points:
            if crop_time:
                if crop_time.start and point.time < crop_time.start:
                    continue
                if crop_time.end and point.time >= crop_time.end:
                    break
            if in_points_counter == self._opts.get('limit'):
                log.debug("Limit of processed trackpoints ({}) reached, ending.".format(self._opts.get('limit')))
                break
            
            duplicated_point = None
            hr_info = ""
            current_speed_ms = 0
            if prev_point:
                length_from_prev = geo.length_3d((prev_point, point))
            if pace:
                if length_from_prev > 0:
                    # compute the arrival time to this point
                    next_time = pace_last_time + datetime.timedelta(
                        seconds = round(length_from_prev / speed_mps))
                    pace_last_time = min(next_time, fill_time.end)
                    point.time = pace_last_time
                    # add the waiting time (proportionally from pause_time) to new duplicated point
                    if current_length_for_pause >= PAUSE_LIMIT_METERS:
                        pause_seconds = round(pause_time * ((1.0*current_length_for_pause)/total_length))
                        next_time = pace_last_time + datetime.timedelta(seconds = pause_seconds)
                        pace_last_time = min(next_time, fill_time.end)
                        elapsed_pause_time += pause_seconds
                        log.debug('Setting waiting time: {}s. Total pause time: {}s, already set: {}s'
                            .format(pause_seconds, pause_time, elapsed_pause_time))
                        duplicated_point = deepcopy(point)
                        duplicated_point.time = pace_last_time
                        current_length_for_pause = 0
                    else:
                        current_length_for_pause += length_from_prev
                else:
                    point.time = pace_last_time
            elif not point.time:
                raise StravaGpxException('No pace was set, but there is no time for the point: {}'.format(point))

            if hr is not None or hr_points is not None:
                if point.extensions and not self._opts.get('soft'):
                    raise StravaGpxException("Existing *extension* value found."+
                    "Consider running with --soft parameter. Value found: {}".
                    format(point.extensions))
                extension_element = ElementTree.Element(HR_BASE_TAG)
                extension_element.text = ""
                hr_element = ElementTree.Element(HR_TAG)
                hr_value = str(self.get_hr_for_time(point.time) or hr)
                hr_element.text = hr_value
                extension_element.append(hr_element)
                point.extensions.append(extension_element)
                hr_info = ", HR: {}".format(hr_value)
            
            total_distance += length_from_prev
            self.add_point(point)
            if prev_point:
                if point.time == prev_point.time:
                    log.warn('Point time ({}) has not changed since the previous point (and it is not duplicated stop point)! Length from previous: {}m'
                        .format(point.time, length_from_prev))
                else:
                    current_speed_ms = (length_from_prev / (point.time - prev_point.time).seconds)
            log.debug("Added point at time: {}, current total distance: {:.1f} m, from previous: {:.1f} m, current speed {:.2f} km/h{}"
                .format(point.time, total_distance, length_from_prev, current_speed_ms*3.6, hr_info))
            if duplicated_point:
                log.debug(">> Added duplicated (stop) point at time {}".format(duplicated_point.time))
                self.add_point(duplicated_point)
                out_points_counter += 1
                prev_point = duplicated_point
            else:
                prev_point = point
            in_points_counter += 1
            out_points_counter += 1
        
        return (out_points_counter, total_distance)
    
    def get_hr_for_time(self, search_time):
        """Get HR value (from HR file) corresponding to the given time.
        """
        offset = self._time_hr_search_offset
        if len(self._time_hr_array) and offset < len(self._time_hr_array):
            if search_time < self._time_hr_array[offset][0]:
                return self._time_hr_array[offset][1]
            while search_time >= self._time_hr_array[offset][0]:
                if offset == len(self._time_hr_array):
                    raise StravaGpxException("No corresponding datetime found in HR file.")
                self._time_hr_search_offset += 1
                return self._time_hr_array[offset][1]
        return None
    
    def process(self):
        """Processes the action based on selected mode.
        Prepares the output file and does the XML postprocessing and write.
        """
        # prepare the output
        out_gpx = gpx.GPX()
        out_track = gpx.GPXTrack()
        out_gpx.tracks.append(out_track)
        self._out_segment = gpx.GPXTrackSegment()
        out_track.segments.append(self._out_segment)

        if self._opts['mode'] == 'fill':
            self.fill()
        elif self._opts['mode'] == 'merge':
            self.merge()
        elif self._opts['mode'] == 'fix':
            self.fix()
        elif self._opts['mode'] == 'crop':
            self.crop()
        else:
            raise StravaGpxException("Invalid processing mode")
        
        # postprocessing - namespace
        out_gpx.nsmap['gpxtpx'] = GARMIN_NS
        
        try:
            with open(self._opts['output'], 'w') as stream:
                stream.write(out_gpx.to_xml())
        except IOError as e:
            raise StravaGpxException("Error while writing the output XML to file: {}".
                format(self._opts['output']))
        log.info("Output GPX file written to \"{}\"".format(self._opts['output']))
    
    def fill(self):
        """Performs the fill mode processing.
        Adds time and/or heart rate values.
        """
        pace = StravaGpxTool.parse_pace(self._opts.get('pace')) if self._opts.get('pace') else None
        hr = self._opts.get('hr')
        hr_points = None
        fill_time = start_time = end_time = duration_total = None

        # validations and variables settings
        if not pace and not hr:
            raise StravaGpxException("ERROR: Nothing to set (no HR or pace specified in program parameters)")
        if pace:
            if not self._opts.get('start_time') or not self._opts.get('end_time'):
                raise StravaGpxException("\"start-time\" and \"end-time\" arguments must be set for filling the pace.")
            if self._opts.get('start_time') >= self._opts.get('end_time'):
                raise StravaGpxException("End date to set is not higher than start date.")
            # process start and end dates
            try:
                start_time = dateutil.parser.parse(self._opts.get('start_time'))
            except ValueError as e:
                raise StravaGpxException("Invalid \"start_time\" parameter: {}".format(self._opts.get('start_time')))
            try:
                end_time = dateutil.parser.parse(self._opts.get('end_time'))
            except ValueError as e:
                raise StravaGpxException("Invalid \"end_time\" parameter: {}".format(self._opts.get('end_time')))
            fill_time = PointsBoundaries(start_time, end_time)

        # read input files
        in_points = StravaGpxTool.get_points_from_file(self._opts['input'])
        # HR input file
        if self._opts['hr_file']:
            hr_points = StravaGpxTool.get_points_from_file(self._opts['hr_file'])

        # process all points from input file
        metrics = self.add_points(in_points, hr, hr_points, pace, fill_time)
        log.info("Total distance: {}".format(metrics[1]))
    
    def merge(self):
        """Performs the merge mode processing.
        Adds all points from the both input files.
        """
        limit = self._opts.get('limit')
        if not os.path.isdir(self._opts['input-dir']):
            raise StravaGpxException("Path \"{}\" is not valid directory for input files."
                .format(self._opts['input-dir']))
        input_files = []
        try:
            input_files = [f for f in os.listdir(self._opts['input-dir'])
                if os.path.isfile(os.path.join(self._opts['input-dir'], f))]
        except OSError as e:
            raise StravaGpxException("Could not read files from input files directory \"{}\"."
                .format(self._opts['input-dir']))
        input_files.sort()
        log.info("Merging files from \"{}\" directory: {}."
            .format(self._opts['input-dir'], ", ".join(input_files)))

        for filename in input_files:

            in_points = StravaGpxTool.get_points_from_file(os.path.join(self._opts['input-dir'], filename))

            i = 0
            for point in in_points:
                self.add_point(point)
                i += 1
                if i == limit:
                    log.debug("Limit of processed trackpoints ({}) reached, ending.".format(limit))
                    break
    
    def fix(self):

        start_distance = int(self._opts['start-distance']) # meters
        end_distance = int(self._opts['end-distance'])  # meters
        pace = StravaGpxTool.parse_pace(self._opts.get('pace'))

        # read input files
        in_points = StravaGpxTool.get_points_from_file(self._opts['input'])
        correction_points = StravaGpxTool.get_points_from_file(self._opts['correction-file'])

        # get indicies of points in correction file
        def get_fix_indexes(distance, search_from_end=False):
            dst_point = StravaGpxTool.get_point_at_distance(in_points, distance)
            log.debug('Found the split point at time: {}'.format(dst_point.time))
            return (
                StravaGpxTool.get_closest_point_index(in_points, dst_point),
                StravaGpxTool.get_closest_point_index(correction_points, dst_point, search_from_end)
            )
        in_start_index, fix_start_index = get_fix_indexes(start_distance)
        log.debug('Split points start index - from input file {} (at time: {})'
            .format(in_start_index, in_points[in_start_index].time))
        in_end_index, fix_end_index = get_fix_indexes(end_distance, True)
        log.debug('Split points end indexes - from input file {} (at time: {})'
            .format(in_end_index, in_points[in_end_index].time))
        fill_time_boundaries = PointsBoundaries(
            StravaGpxTool.get_time_at_index(in_points, in_start_index-1), # last inserted point
            StravaGpxTool.get_time_at_index(in_points, in_end_index))
        # correction in corrected indexes to smooth the path - use the next points
        fix_start_index += 1
        fix_end_index -= 1
        
        # add point from input file from begining to the first fixed point
        metrics = self.add_points(in_points, crop_index=PointsBoundaries(None, in_start_index))
        log.info('Part before fix to distance {} m: added {} points with distance {:.0f} m.'.format(start_distance, metrics[0], metrics[1]))
        # add correction (fix) points from correction file
        metrics = self.add_points(correction_points, hr=None, hr_points=in_points,
            pace=pace, fill_time=fill_time_boundaries, end_next_point=in_points[in_end_index],
            crop_index=PointsBoundaries(fix_start_index, fix_end_index),
            next_point=in_points[in_end_index])
        log.info('Fixed part: added {} points with distance {:.0f} m to the time between {} and {}'.format(metrics[0], metrics[1], fill_time_boundaries.start, fill_time_boundaries.end))
        # add points from input file from the last fixed point to the end
        metrics = self.add_points(in_points, crop_index=PointsBoundaries(in_end_index, None))
        log.info('Part after fix from distance {}: added {} points with distance {:.0f} m.'.format(end_distance, metrics[0], metrics[1]))

def main():
    """Main program entrypoint"""
    # program arguments
    parser = argparse.ArgumentParser(description="Tool for manipulating GPX files for Strava.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--output', default='out.gpx', help="Output file")
    parser.add_argument('--debug', action='store_true', help="Turn on debug mode")
    parser.add_argument('--limit', type=int, help="DEBUG argument: Limit the number of trackpoint to process")
    # program mode
    mode_subpars = parser.add_subparsers(dest='mode')
    mode_subpars.required = True
    # subparsers args
    merge_subpar = mode_subpars.add_parser('merge', help="Merges GPX files from the input directory")
    merge_subpar.add_argument('input-dir', help="Path to directory with input files")
    fill_subpars = mode_subpars.add_parser('fill', help="Fills GPX points with missing attributes: time or heart rate")
    fill_subpars.add_argument('input', help="Input GPX file")
    fill_subpars.add_argument('--pace', help="Average pace to set for all points with this value missing. Format: MM:SS")
    fill_subpars.add_argument('--start-time', help="Start time for filling the pace (ISO format, UTC)")
    fill_subpars.add_argument('--end-time', help="End time for filling the pace (ISO format, UTC)")
    fill_subpars.add_argument('--hr', type=int, help="Heart rate to set for all points with this value missing")
    fill_subpars.add_argument('--hr-file', help="File containing heart rate values - they will be added to the input file (based on the time) for all points with this value missing")
    fill_subpars.add_argument('--soft', action='store_true', help="Soft mode - it fills just missig values and ignores existing")
    fix_subpar = mode_subpars.add_parser('fix', help="Fix corrupted path in GPX file")
    fix_subpar.add_argument('input', help="Input GPX file")
    fix_subpar.add_argument('correction-file', help="GPX file with corrected path segment")
    fix_subpar.add_argument('start-distance', type=float, help="Start distance from the begining for correction in meters")
    fix_subpar.add_argument('end-distance', type=float, help="End distance from the begining for correction in meters")
    fix_subpar.add_argument('pace', help="Average pace to set for all points with this value missing. Format: MM:SS")

    opts = vars(parser.parse_args())

    # logging level based on flag from arguments
    log.setLevel("DEBUG" if opts['debug'] else "INFO")

    try:
        processor = StravaGpxTool(opts)
        processor.process()
    except StravaGpxException as e:
        log.error("Error while processing: {}".format(e))
        sys.exit(os.EX_SOFTWARE)


if __name__ == '__main__':
    main()