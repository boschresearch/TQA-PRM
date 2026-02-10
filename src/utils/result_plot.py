""" Utility classes and functions related to TQA-PRM (EACL 2026).
Copyright (c) 2026 Robert Bosch GmbH
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
butWITHOUT ANYWARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""


import matplotlib.pyplot as plt

# X-axis values: number of action generations
def main():
    x = [1, 3, 5, 10]

    # Example EM values (replace with your actual numbers)
    wtq =  [71.5, 72.4, 72.7, 73.1]
    tat =  [64.8, 65.9, 66.2, 67.6]
    crt =  [58.2, 62.1, 64.4, 65.6]
    scitab = [56.0, 57.2, 59.8, 59.6]

    plt.figure(figsize=(6, 4.5), dpi=150)

    plt.plot(x, wtq, marker='o', linewidth=2, label='WTQ')
    plt.plot(x, tat, marker='o', linewidth=2, label='TAT')
    plt.plot(x, crt, marker='o', linewidth=2, label='CRT')
    plt.plot(x, scitab, marker='o', linewidth=2, label='SCITAB')

    plt.xlabel('Number of action generation')
    plt.ylabel('EM')
    plt.xticks(x)
    plt.ylim(55, 75)
    plt.legend(loc='upper left', frameon=True)
    plt.tight_layout()

    print("About to show the figure… (this should block)")
    plt.show(block=True)
if __name__ =="__main__":
    main()
