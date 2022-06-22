import pprint
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt


def main():
    results = pd.read_csv(Path(".").resolve() / "results.csv")

    cut = "TargetModule"
    config = "ConfigurationId"
    project_name = "ProjectName"
    coverage = "Coverage"
    seconds = [f"CoverageTimeline_T{i}" for i in range(1, 601)]
    runs = 30

    counts = results[cut].value_counts()
    results = results[results[cut].isin(counts[counts == runs * 5].index)]

    number_cuts = len(set(results[cut]))
    print(
        f"I tested {number_cuts} unique classes, each being executed {runs} times per "
        f"configuration"
    )

    no_deeptyper = results[results[config] == "no_deeptyper"]
    rt0_rndch = results[results[config] == "rt0_rndch"]
    rt01_rndch = results[results[config] == "rt0.1_rndch"]
    rt02_rndch = results[results[config] == "rt0.2_rndch"]
    rt03_rndch = results[results[config] == "rt0.3_rndch"]

    config_names = list(set(results[config]))
    config_names.sort()
    print("I used {} configurations, namely:\n - {}".format(
        len(config_names), "\n - ".join(config_names)
    ))

    konfigs = {}
    for name in config_names:
        konfigs[name] = results[results[config] == name]

    data_points = {}
    executed_classes = {}
    failing_classes = {}
    print("Available Data Points:")
    print("----------------------")
    for k in konfigs:
        l = len(set((konfigs[k])[cut]))
        executed_classes[k] = set((konfigs[k])[cut])
        failing_classes[k] = set(results[cut]).difference(executed_classes[k])
        data_points[k] = l
        s = " --> {:> 4} CUTS failed entirely".format(
            number_cuts - l
        ) if l != number_cuts else ""
        print("{: >30}: {: >4} / {} CUTS tested".format(k, l, number_cuts), s)

    #print("")
    #print("")
    all_classes = set.intersection(*[v for _, v in executed_classes.items()])
    #print(f"Passing classes: {len(all_classes)}")
    #print(all_classes)

    #print(f"Failing classes: {len(failing_classes)}")
    #pprint.pprint(failing_classes)

    toplot = {}
    legends = [
        "Baseline",
        "RT0",
        "RT10",
        "RT20",
        "RT30",
    ]
    for k, cfg in enumerate([
        no_deeptyper,
        rt0_rndch,
        rt01_rndch,
        rt02_rndch,
        rt03_rndch,
        
    ]):
        toplot[legends[k]] = (cfg[
                                  cfg[cut].isin(all_classes)
                              ][seconds].mean())
    #timestamps = [0,100,200,300,400,500,599]
    #for t in timestamps:
        #print("\n")
        #print(toplot["random-type-0.1"][t]-toplot["no-deeptyper"][t])
        #for l in legends:
            #print(l, toplot[l][t])
            
    fig, ax = plt.subplots()
    pd.DataFrame(toplot).plot(ax=ax, figsize=(10, 5))
    ax.set_xticks(range(0, 601, 30))
    ax.set_xticklabels([i for i in range(0, 601, 30)])
    ax.set_ylabel("Coverage")
    ax.set_xlabel("Time (s)")
    plt.savefig(PAPER_EXPORT_PATH / "." / "coverage-over-time.pdf", bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    main()
