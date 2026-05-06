#!/usr/bin/env python3

from netCDF4 import Dataset
from os import path
from glob import glob
import numpy as np
import argparse
import sys
from math import ceil

# This script extracts the information from the [point|surface|volume]_data_*.Netcdf files which are
# produced by the point/surface/volume data sampling feature. You can select various ranges of
# interest. The results are written to stdout or a file

def main():
    class formatter(argparse.ArgumentDefaultsHelpFormatter,
            argparse.RawDescriptionHelpFormatter):
        pass
    parser = argparse.ArgumentParser(description="Processes files produced by "
            "the point/surface/volume data sampling feature.\nProvide all ranges "
            "seperated by a colon (eg. 3:5.2).\nYou can also pass a single "
            "value.\nYou can either select via timestep range or via the "
            "time range,\nbut not via both.\n\nExample:\nExtract recorded "
            "states from points 5 to 20\nfrom time range 0.7 to 2.1 seconds.\n"
            "Files are at ~/scratch/stuff/out:\n\n"
            "  ./process_point_data.py -i ~/scratch/stuff/out -t 0.7:2.1 -p 5:20",
            formatter_class=formatter)
    parser.add_argument("-i", "--input", dest="input_path",
            help="input directory", default=".")
    parser.add_argument("-f", "--file-prefix", dest="file_prefix",
            help="file prefix", default="point_data_")
    parser.add_argument("-n", "--filenumber", dest="filenumber", help="number of "
            "input file", default="0")
    parser.add_argument("-p", "--points", dest="input_points", help="input "
            "points index range, starts at 1 (e.g. \"3\" or \"3:10\")",
            default="all")
    parser.add_argument("-s", "--timestep-range", dest="timestep_range",
            help="select range of timesteps to output(e.g. \"3\" or \"3:10\")",
            default="all")
    parser.add_argument("-t", "--time-range", dest="time_range", help="select "
            "range of simulation time to output(e.g. \"3.2\" or \"3:10.1\")",
            default="all")
    parser.add_argument("-o", "--output", help="output file path",
            default="stdout")
    args = parser.parse_args()

    if args.timestep_range != "all" and args.time_range != "all":
        print("ERROR: Please select a range to process either via timesteps or "
                "time, but not via both.", file=sys.stderr)
        exit(1)

    # get relevant files
    file_string = "".join((args.file_prefix, args.filenumber, "_*Netcdf"))
    file_names = glob(path.join(args.input_path, file_string))
    if len(file_names) == 0:
        print("ERROR: No sampling data files found. Exiting.", file=sys.stderr)
        exit(1)

    nc_files = [ Dataset(f, "r") for f in sorted(file_names) ]

    # parse point index input
    point_start = False
    point_end = False
    if args.input_points != "all":
        if args.input_points.count(":") == 1:
            point_start, point_end = args.input_points.split(":")
            point_start = int(point_start)
            point_end = int(point_end)
        else:
            point_start = int(args.input_points)
            point_end = point_start
    else:
        point_start = 1 # so line number in input file == this
        point_end = len(nc_files[0].variables["sortIndex"])

    if point_end < point_start:
        print("ERROR: Point index range end is larger than point index range "
            "start!", file=sys.stderr)
        exit(1)
    if point_start < 1:
        print("ERROR: Point index of less than 1 is invalid!", file=sys.stderr)
        exit(1)
    if point_end > len(nc_files[0].variables["sortIndex"]):
        print("ERROR: Point index range end is larger than there are points!",
            file=sys.stderr)
        exit(1)

    # get selected range
    range_selection = True
    if args.time_range == "all" and args.timestep_range == "all":
        range_selection = False
    range_start = -1
    range_end = -1
    selected_range = (args.time_range if args.time_range != "all"
        else args.timestep_range)

    if selected_range.count(":") == 1:
        range_start, range_end = selected_range.split(":")
    elif range_selection:
        range_start = selected_range

    select_by_timesteps = False

    if args.timestep_range != "all":
        select_by_timesteps = True
        range_start = int(range_start)
        if (range_end):
            range_end = int(range_end)

    if args.time_range != "all":
        select_by_timesteps = False
        range_start = float(range_start)
        if (range_end):
            range_end = float(range_end)

    save_index_start = -1
    save_index_end = -1

    # get save indices where to start and stop
    save_index = 0

    def equal(a, b):
        if select_by_timesteps:
            return a == b
        else:
            return np.isclose(a, b)

    for nc_file in nc_files:
        range_ = None
        if select_by_timesteps:
            range_ = nc_file.variables["timeStep"]
        else:
            range_ = nc_file.variables["time"]
        for i in range_:
            if i >= range_start and save_index_start == -1:
                save_index_start = save_index
            if i > range_end and save_index_end == -1:
                save_index_end = save_index - 1
            if equal(i, range_end) and save_index_end == -1:
                save_index_end = save_index
            save_index += 1

    if save_index_start == -1 and range_selection:
        print("WARNING: Could not find selected beginning. Selecting start of "
              "recorded data.")
        save_index_start = 0
    if save_index_end == -1 and range_end != -1 and range_selection:
        print("WARNING: Could not find selected end. Selecting end of recorded "
              "data.")
        save_index_end = save_index - 1
    elif range_end == -1 and range_selection:
        save_index_end = save_index_start
    if not range_selection:
        save_index_start = 0
        save_index_end = save_index - 1

    # get cons var names from attributes
    var_names = [ v for v in nc_files[0].variables["pointStates"].ncattrs()
            if "var_" in v ]
    var_names = list(map(
            lambda x: nc_files[0].variables["pointStates"].getncattr(x),
            var_names))

    # get relevant reverse indices
    # point_start-1 because line number input file == selected equivalent
    reverse_index = []
    for n, i in enumerate(nc_files[0].variables["sortIndex"]):
        if i in range(point_start-1, point_end):
            reverse_index.append((n, i))
    reverse_index = [ x[0] for x in sorted(reverse_index, key=lambda x: x[1]) ]

    # gather information about save indices
    files_with_meta = list()
    offset = 0
    for nc_file in nc_files:
        length = len(nc_file.dimensions[nc_file.variables["pointStates"]
            .dimensions[1]])
        files_with_meta.append((nc_file, length, offset))
        offset += length

    # write out requested data
    with (open(args.output, "w") if args.output != "stdout"
            else sys.stdout) as output:
        output.write("time_step time ")
        for i in range(point_start, point_end + 1):
            for var in var_names:
                output.write("{}_p{} ".format(var, i))
        output.write("\n")
        no_samples = save_index_end - save_index_start + 1
        sample_no = 0
        for nc_file, length, offset in files_with_meta:
            for i in range(length):
                if (save_index_start <= (i + offset)
                    and save_index_end >= (i + offset)):
                    timestep = nc_file.variables["timeStep"][i]
                    time = nc_file.variables["time"][i]
                    output.write("{} ".format(str(timestep)))
                    output.write("{} ".format(str(time)))
                    for n, j in enumerate(reverse_index):
                        for k in range(len(var_names)):
                            output.write("{} ".format(
                                nc_file.variables["pointStates"][j, i, k]))
                        percentage = (sample_no/no_samples
                                + n/len(reverse_index)/no_samples)
                        write_progress_bar(output, percentage)
                    output.write("\n")
                    sample_no += 1

    # write final newline after progress bar
    if (args.output != "stdout"):
        print()

def write_progress_bar(handle, percentage):
    """writes progress bar if appropriate"""
    if handle != sys.stdout and sys.stdout.isatty():
        progress = ceil(percentage * 70)
        sys.stdout.write("\rProgress: [{}{}]".format("." * progress,
            " " * (70 - progress)))

if __name__ == "__main__":
    main()
