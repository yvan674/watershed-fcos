"""GPU Usage.

Outputs GPU usage in a Line chart along with memory usage in a bar chart.


Author:
    Yvan Satyawan <y_satyawan@hotmail.com>

Created on:
    December 13, 2019
"""
from curses import newwin, KEY_RESIZE, doupdate
import curses
from datetime import datetime, timedelta
from time import sleep
import GPUtil
from math import ceil, floor
import sys
import argparse


def parse_argument():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description='graphically show GPU usage')

    parser.add_argument('-i', '--interval', type=float,
                        help='update interval in seconds')

    return parser.parse_args()


def plot(series: list or tuple, cfg: dict = None):
    """Returns lines of an ascii plot.

    Notes:
        Possible cfg parameters are 'minimum', 'maximum', 'offset', 'height'
        and 'format'.

    Example:
        >>> series = [2, 5, 1, 3, 4, 1]
        >>> print(plot(series, { 'height' :10 }))
    """
    cfg = cfg or {}
    minimum = cfg['minimum'] if 'minimum' in cfg else min(series)
    maximum = cfg['maximum'] if 'maximum' in cfg else max(series)

    interval = abs(float(maximum) - float(minimum))
    offset = cfg['offset'] if 'offset' in cfg else 3
    height = cfg['height'] if 'height' in cfg else interval
    ratio = height / interval
    min2 = floor(float(minimum) * ratio)
    max2 = ceil(float(maximum) * ratio)

    intmin2 = int(min2)
    intmax2 = int(max2)

    rows = abs(intmax2 - intmin2)
    width = len(series) + offset
    placeholder = cfg['format'] if 'format' in cfg else '{:8.2f} '

    result = [[' '] * width for _ in range(rows + 1)]

    # axis and labels
    for y in range(intmin2, intmax2 + 1):
        label = placeholder.format(
            float(maximum) - ((y - intmin2) * interval / rows))
        result[y - intmin2][max(offset - len(label), 0)] = label
        result[y - intmin2][offset - 1] = '┼' if y == 0 else '┤'

    y0 = int(series[0] * ratio - min2)
    result[rows - y0][offset - 1] = '┼'  # first value

    for x in range(0, len(series) - 1):  # plot the line
        y0 = int(round(series[x + 0] * ratio) - intmin2)
        y1 = int(round(series[x + 1] * ratio) - intmin2)
        if y0 == y1:
            result[rows - y0][x + offset] = '─'
        else:
            result[rows - y1][x + offset] = '╰' if y0 > y1 else '╭'
            result[rows - y0][x + offset] = '╮' if y0 > y1 else '╯'
            start = min(y0, y1) + 1
            end = max(y0, y1)
            for y in range(start, end):
                result[rows - y][x + offset] = '│'

    return [''.join(row) for row in result]


def main(stdscr, update_interval: float = 1.):
    # Clear screen
    stdscr.clear()
    stdscr.nodelay(True)

    last_screen_size = (0, 0)

    last_update = datetime.now()
    gpus = GPUtil.getGPUs()

    if len(gpus) == 0:
        raise Exception("No GPUs found")

    val_utilizations = []
    windows = []
    sizes = -1

    while True:
        # Handle window resize
        if stdscr.getmaxyx() != last_screen_size:
            last_screen_size = stdscr.getmaxyx()
            stdscr.clear()
            # calculate windows
            sizes = calculate_sizes(stdscr, len(gpus))
            windows = []
            if sizes == -1:
                # This is an error meaning that the terminal window is too small
                stdscr.clear()
                h, w = stdscr.getmaxyx()
                error_string = ['Window is too small.',
                                'Please make it bigger',
                                'or press "Q" to quit.']
                assert w > 21, 'Terminal window is too small.'
                for i in range(len(error_string)):
                    stdscr.addstr(i, 0, error_string[i])
                stdscr.refresh()
                stdscr.nodelay(False)
                key = stdscr.getkey()
                if key == 'q':
                    sys.exit()
                elif key == KEY_RESIZE:
                    stdscr.nodelay(True)
                    continue
            else:
                for i in range(len(gpus)):
                    win = newwin(sizes[i]['nlines'], sizes[i]['ncols'],
                                 sizes[i]['begin_y'], sizes[i]['begin_x'])
                    win.clear()
                    win.border()
                    win.addstr(0, 2, "GPU {}: {}".format(i, gpus[i].name))
                    win.noutrefresh()
                    windows.append(win)
                    val_utilizations.append([0, 0])

        if datetime.now() - last_update > timedelta(seconds=update_interval) \
                and sizes != -1:
            gpus = GPUtil.getGPUs()
            for i in range(len(gpus)):
                # Get utilizations
                val_utilizations[i].append(gpus[i].load * 100)

                # Create plot for GPU utilization
                util_window = windows[i].derwin(sizes[i]['nlines'] - 4,
                                                sizes[i]['ncols'] - 9,
                                                2,
                                                2)

                val_utilizations[i] = draw_utilization_plot(util_window,
                                                            val_utilizations[i])

                # Create bar chart for memory utilization
                mem_window = windows[i].derwin(
                    sizes[i]['nlines'] - 2, 5, 2, sizes[i]['ncols'] - 7)

                draw_bar_chart(mem_window, gpus[i])

                # Add borders a gain and put the window title
                windows[i].border()
                windows[i].addstr(0, 2, "GPU {}: {}".format(i, gpus[i].name))
                windows[i].noutrefresh()
            last_update = datetime.now()
            doupdate()
        sleep(update_interval / 2)


def draw_utilization_plot(window, values: list) -> list:
    """Draws the GPU utilization plot.

    Args:
        window: Window to draw the chart in.
        values: The values to draw on the chart.

    Returns:
        The utilization list, truncated if it's
    """
    h, w = window.getmaxyx()
    if len(values) > w - 7:
        values.pop(0)

    res = plot(values, cfg={'minimum': 0, 'maximum': 100,
                            'height': h - 1, 'format': '{:3.0f}%',
                            'offset': 3})

    for i, line in enumerate(res):
        window.addstr(i, 0, line)
    window.noutrefresh()
    return values


def draw_bar_chart(window, gpu: GPUtil.GPUtil.GPU):
    """Draws a bar chart.

    Args:
        window: Window to draw the chart in
        gpu: The GPU object to draw a memory bar chart of.
    """
    window.clear()
    h, w = window.getmaxyx()
    total = gpu.memoryTotal
    value = gpu.memoryUsed

    # Calculate number of blocks to use. Each row can either be 1 or 2 blocks.
    blocks = round(value / total * (h + h - 2))
    # If this requires a half block, the value will be odd
    full_rows = floor(blocks / 2)
    half_row = False if blocks % 2 == 0 else True

    # Draw the label
    value = '{:5.0f}'.format(value)
    # First, center it
    start_pos = int((w / 2)) - int((len(value) / 2))
    window.addstr(h - 2, start_pos, value)
    # Now draw in the rows
    for i in range(h - 3, -1, -1):
        if full_rows == 0 and not half_row:
            break
        if full_rows != 0:
            window.addstr(i, 0, '█' * w)
            full_rows -= 1
        elif half_row:
            window.addstr(i, 0, '▄' * w)
            half_row = False

    window.noutrefresh()


def calculate_sizes(stdscr, num_gpus: int) -> list or int:
    """Calculate appropriate plot sizes.

    Calculates the sizes that should be appropriate to show plots. If the
    terminal window is large enough, it uses multiple columns. It returns the
    boxes for each GPU window

    Returns:
        A list of 4-tuples containing (x, y, w, h), or -1 if the window is
        too small.
    """
    out_dict = []
    h, w = stdscr.getmaxyx()
    w -= 1
    h -= 1

    # Figure out appropriate layout
    min_height = 10
    min_width = 35

    columns = floor(w / min_width)
    rows = ceil(num_gpus / columns)

    width = floor(w / columns)
    height = floor(h / rows)

    if width < min_width or height < min_height:
        return -1

    for i in range(num_gpus):
        col = i % columns
        row = floor(i / columns)

        x_pad = 1 if col > 0 else 0
        y_pad = 1 if row > 0 else 0

        x = (col * width) + x_pad
        y = (row * height) + y_pad

        out_dict.append({'nlines': height, 'ncols': width, 'begin_y': y,
                         'begin_x': x})

    return out_dict


if __name__ == '__main__':
    args = parse_argument()
    try:
        # Initialize curses
        stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        stdscr.keypad(1)

        try:
            curses.start_color()
        except:
            pass
        if args.interval:
            main(stdscr, args.interval)
        else:
            main(stdscr)
    except KeyboardInterrupt:
        pass
    finally:
        # Set everything back to normal
        if 'stdscr' in locals():
            stdscr.keypad(0)
            curses.echo()
            curses.nocbreak()
            curses.endwin()
