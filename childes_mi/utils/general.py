import matplotlib.pyplot as plt

import collections

def readfile(file):
    with open(file, 'r') as file:
        data = file.read()#.replace('\n', '')
    return data

#flatten = lambda l: [item for sublist in l for item in sublist]
def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


def save_fig(loc):
	plt.savefig(str(loc)+'.pdf',dpi=300, bbox_inches = 'tight',
    pad_inches = 0)
	plt.savefig(str(loc)+'.png',dpi=300, bbox_inches = 'tight',
	    pad_inches = 0)
	plt.savefig(str(loc)+'.svg',dpi=300, bbox_inches = 'tight',
	    pad_inches = 0)
	plt.savefig(str(loc)+'.png',dpi=300, bbox_inches = 'tight',
	    pad_inches = 0)
	plt.savefig(str(loc)+'.jpg',dpi=150, bbox_inches = 'tight',
	    pad_inches = 0)