import os

def writeBags(bagsDir, examplesFile, bagsFile):
    examples = open(examplesFile).readlines()

    # make example-bag map by reading the dpd_output directory
    ex_bags = {}
    for bagid in os.listdir(bagsDir):
        if bagid.find('_') == -1:
            continue
        parts = bagid.split('_')
        print(parts)
        exid, bagid = parts[0], parts[1].split('.')[0]
        if exid not in ex_bags:
            ex_bags[exid] = []
        ex_bags[exid].append(bagid)

    # make the list of bag examples
    bags = []
    for ex in examples:
        exid = ex.split(' ')[2][:-1]
        if exid not in ex_bags:
            continue
        for bagid in ex_bags[exid]:
            bag = ex[:ex.find(')')] + '_' + bagid + ex[ex.find(')'):]
            bags.append(bag)

    # write bag exmaples file
    with open(bagsFile, 'w') as outf:
        for bag in bags:
            outf.write(bag)

root = '/scratch/cse/dual/cs5130298/dpd/miml/'
for x in ['pos_20', 'pos_40', 'pn_20', 'pn_40']:
    dpdDir = root + 'dpd_postpruned_' +  x + '/'
    print(dpdDir)
    for exFile in ['random-split-1-train-100', 'random-split-1-train']:
        writeBags(dpdDir, root + exFile + '.examples', dpdDir + exFile + '-bags.examples')