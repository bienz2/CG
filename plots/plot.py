import os

folder = "../results/dane"

class CGTiming():

    def __init__(self):
        self.times = list()
        self.avg = 0
        self.high = 0
        self.low = 0

    def add_timing(self, time):
        self.times.append(time)

    def average(self):
        self.avg = 0
        for t in self.times:
            self.avg += t
        self.avg = self.avg / len(self.times)
        self.high = max(self.times) - self.avg
        self.low = self.avg - min(self.times)

class CGTest:

    def __init__(self):
        self.pmpi = CGTiming()
        self.mpil_rd = CGTiming()
        self.mpil_na = CGTiming()
        self.mpil_la = CGTiming()
        self.mpil_radix = CGTiming()
        self.mpil_rd_pers = CGTiming()
        self.mpil_na_pers = CGTiming()
        self.mpil_la_pers = CGTiming()
        self.mpil_radix_pers = CGTiming()
        self.mpil_rma_hier = CGTiming()
        self.mpil_rma_hier_eb = CGTiming()

    def average(self):
        self.pmpi.average()
        self.mpil_rd.average()
        self.mpil_na.average()
        self.mpil_la.average()
        self.mpil_radix.average()
        self.mpil_rd_pers.average()
        self.mpil_na_pers.average()
        self.mpil_la_pers.average()
        self.mpil_radix_pers.average()
        self.mpil_rma_hier.average()
        self.mpil_rma_hier_eb.average()


class AdvanceTest:

    def __init__(self, nodes):
        self.nodes = nodes
        self.standard = CGTest()
        self.persistent = CGTest()
        self.locality = CGTest()
        self.persistent_loc = CGTest()

    def average(self):
        self.standard.average()
        self.persistent.average()
        self.locality.average()
        self.persistent_loc.average()

class Matrix:

    def __init__(self, name):
        self.name = name
        self.nodes = list()
        self.tests = list()

    def add_node(self, nodes):
        self.nodes.append(nodes)
        self.tests.append(AdvanceTest(nodes))

    def sort(self):
        paired = sorted(zip(self.nodes, self.tests), key=lambda x: x[0])
        self.nodes, self.tests = map(list, zip(*paired))

    def average(self):
        for test in self.tests:
            test.average()

matrices = list()
advance = ""

for filename in os.listdir(folder):
    if not filename.endswith(".out"):
        continue

    path = os.path.join(folder, filename)
    if os.path.isfile(path):
        with open(path, "r") as f:
            for line in f:
                if "srun --ntasks-per-node" in line:
                    name = (line.split('/')[-1]).strip(".pm\n")

                    parts = line.split()
                    idx = parts.index("-N") + 1;
                    nodes = int(parts[idx])

                    matrix = None
                    for mat in matrices:
                        if mat.name == name:
                            matrix = mat
                    if matrix is None:
                        matrix = Matrix(name)
                        matrices.append(matrix)

                    matrix.add_node(nodes)

                    continue
                
                if "Running with" in line and "Neighbor Collectives" in line:
                    if "Pers Standard" in line:
                        advance = matrix.tests[-1].persistent
                    elif "Pers Locality" in line:
                        advance = matrix.tests[-1].persistent_loc
                    elif "Standard" in line:
                        advance = matrix.tests[-1].standard
                    elif "Locality" in line:
                        advance = matrix.tests[-1].locality

                    continue

                if "CG with" in line:
                    time = float(line.split(': ')[-1])

                    if "PMPI" in line:
                        advance.pmpi.add_timing(time)
                    elif "RMA Hier EB" in line:
                        advance.mpil_rma_hier_eb.add_timing(time)
                    elif "RMA Hier" in line:
                        advance.mpil_rma_hier.add_timing(time)
                    elif "RADIX" in line:
                        if "Pers" in line:
                            advance.mpil_radix_pers.add_timing(time)
                        else:
                            advance.mpil_radix.add_timing(time)
                    elif "LA" in line:
                        if "Pers" in line:
                            advance.mpil_la_pers.add_timing(time)
                        else:
                            advance.mpil_la.add_timing(time)
                    elif "NA" in line:
                        if "Pers" in line:
                            advance.mpil_na_pers.add_timing(time)
                        else:
                            advance.mpil_na.add_timing(time)
                    elif "RD" in line:
                        if "Pers" in line:
                            advance.mpil_rd_pers.add_timing(time)
                        else:
                            advance.mpil_rd.add_timing(time)


## Sort and calculate stats
import matplotlib.pylab as plt
for matrix in matrices:
    matrix.sort()
    matrix.average()


## Plot Neighbor Collectives
for matrix in matrices:
    # Plot CG with PMPI, scaled across nodes, 1 line per advance
    standard = [test.standard.pmpi.avg for test in matrix.tests]
    standard_error = [[test.standard.pmpi.low for test in matrix.tests],
                        [test.standard.pmpi.high for test in matrix.tests]]
    persistent = [test.persistent.pmpi.avg for test in matrix.tests]
    persistent_error = [[test.persistent.pmpi.low for test in matrix.tests],
                        [test.persistent.pmpi.high for test in matrix.tests]]
    locality = [test.locality.pmpi.avg for test in matrix.tests]
    locality_error = [[test.locality.pmpi.low for test in matrix.tests],
                        [test.locality.pmpi.high for test in matrix.tests]]
    persistent_loc = [test.persistent_loc.pmpi.avg for test in matrix.tests]
    persistent_loc_error = [[test.persistent_loc.pmpi.low for test in matrix.tests],
                        [test.persistent_loc.pmpi.high for test in matrix.tests]]
    print(standard)

    plt.errorbar(matrix.nodes, standard, yerr = standard_error, label = "Standard")
    plt.errorbar(matrix.nodes, persistent, yerr = persistent_error, label = "Persistent")
    plt.errorbar(matrix.nodes, locality, yerr = locality_error, label = "Locality")
    plt.errorbar(matrix.nodes, persistent_loc, yerr = persistent_loc_error, label = "Persistent Locality")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Time (Seconds)")
    plt.legend()
    plt.savefig("Neighbor_%s.pdf"%matrix.name)
    plt.clf()


# Plot Locality-Aware Allreduce
for matrix in matrices:
    pmpi = [test.standard.pmpi.avg for test in matrix.tests]
    pmpi_error = [[test.standard.pmpi.low for test in matrix.tests],
                    [test.standard.pmpi.high for test in matrix.tests]]
    rma = [test.standard.mpil_la_pers.avg for test in matrix.tests]
    rma_error = [[test.standard.mpil_la_pers.low for test in matrix.tests],
                    [test.standard.mpil_la_pers.high for test in matrix.tests]]
    rma_eb = [test.standard.mpil_radix_pers.avg for test in matrix.tests]
    rma_eb_error = [[test.standard.mpil_radix_pers.low for test in matrix.tests],
                    [test.standard.mpil_radix_pers.high for test in matrix.tests]]


    plt.errorbar(matrix.nodes, pmpi, yerr = pmpi_error, label = "PMPI")
    plt.errorbar(matrix.nodes, rma, yerr = rma_error, label = "NUMA-Aware")
    plt.errorbar(matrix.nodes, rma_eb, yerr = rma_eb_error, label = "High-Radix")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Time (Seconds)")
    plt.legend()
    plt.savefig("LocalityAllreduce_%s.pdf"%matrix.name)
    plt.clf()


## Plot EarlyBird
for matrix in matrices:
    pmpi = [test.standard.pmpi.avg for test in matrix.tests]
    pmpi_error = [[test.standard.pmpi.low for test in matrix.tests],
                    [test.standard.pmpi.high for test in matrix.tests]]
    rma = [test.standard.mpil_rma_hier.avg for test in matrix.tests]
    rma_error = [[test.standard.mpil_rma_hier.low for test in matrix.tests],
                    [test.standard.mpil_rma_hier.high for test in matrix.tests]]
    rma_eb = [test.standard.mpil_rma_hier_eb.avg for test in matrix.tests]
    rma_eb_error = [[test.standard.mpil_rma_hier_eb.low for test in matrix.tests],
                    [test.standard.mpil_rma_hier_eb.high for test in matrix.tests]]


    plt.errorbar(matrix.nodes, pmpi, yerr = pmpi_error, label = "PMPI")
    plt.errorbar(matrix.nodes, rma, yerr = rma_error, label = "RMA")
    plt.errorbar(matrix.nodes, rma_eb, yerr = rma_eb_error, label = "EarlyBird")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Time (Seconds)")
    plt.legend()
    plt.savefig("EarlyBird_%s.pdf"%matrix.name)
    plt.clf()

## Combined Locality
for matrix in matrices:
    pmpi = [test.standard.pmpi.avg for test in matrix.tests]
    pmpi_error = [[test.standard.pmpi.low for test in matrix.tests],
                    [test.standard.pmpi.high for test in matrix.tests]]
    radix = [test.standard.mpil_radix_pers.avg for test in matrix.tests]
    radix_error = [[test.standard.mpil_radix_pers.low for test in matrix.tests],
                    [test.standard.mpil_radix_pers.high for test in matrix.tests]]
    persistent_loc = [test.persistent_loc.pmpi.avg for test in matrix.tests]
    persistent_loc_error = [[test.persistent_loc.pmpi.low for test in matrix.tests],
                        [test.persistent_loc.pmpi.high for test in matrix.tests]]
    persistent_loc_radix = [test.persistent_loc.mpil_radix_pers.avg for test in matrix.tests]
    persistent_loc_radix_error = [[test.persistent_loc.mpil_radix_pers.low for test in matrix.tests],
                        [test.persistent_loc.mpil_radix_pers.high for test in matrix.tests]]


    plt.errorbar(matrix.nodes, pmpi, yerr = pmpi_error, label = "Standard + PMPI")
    plt.errorbar(matrix.nodes, radix, yerr = radix_error, label = "Standard + High-Radix")
    plt.errorbar(matrix.nodes, persistent_loc, yerr = persistent_loc_error, label = "Pers. Locality + PMPI")
    plt.errorbar(matrix.nodes, persistent_loc_radix, yerr = persistent_loc_radix_error, label = "Pers. Locality + High-Radix")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Time (Seconds)")
    plt.legend()
    plt.savefig("Combined_%s.pdf"%matrix.name)
    plt.clf()
