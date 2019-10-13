import numpy as np
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
myfont = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simsun.ttc')
#c = pd.read_excel("glioblastoma\主题命名-gbm.xlsx")
#raw_name = list(c["命名"])
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw,shrink=.45, pad=.01)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels,fontsize=10,fontproperties = myfont)
    ax.set_yticklabels(row_labels,fontsize=8,fontproperties = myfont)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im,cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Arguments:
        im         : The AxesImage to be labeled.
    Optional arguments:
        data       : Data used to annotate. If None, the image's data is used.
        valfmt     : The format of the annotations inside the heatmap.
                     This should either use the string format method, e.g.
                     "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
        textcolors : A list or array of two color specifications. The first is
                     used for values below a threshold, the second for those
                     above.
        threshold  : Value in data units according to which the colors from
                     textcolors are applied. If None (the default) uses the
                     middle of the colormap as separation.

    Further arguments are passed on to the created text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[im.norm(data[i, j]) > threshold])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw,fontsize=7)
            texts.append(text)

    return texts

def createheadmap(jou_num):

    jou_num = 20

    onlyFirst = True
    df = pd.read_csv("newdata\\thyroid-cancer__YearModified.csv",sep=',',encoding="utf8")
    from collections import  Counter,OrderedDict
    tpcname = [str(i) for i in range(50)]
    C_tpc = OrderedDict(sorted(Counter(tpcname).items(),key=lambda t: -t[1]))
    Agency = list(df.grantAgency)
    Agency= [str(a).split("|")  for a in Agency if str(a) != "nan"]

    found = []
    for a in Agency:
        found.extend(a)
    if onlyFirst:
        found_ = found
        found = []
        for i in found_:
            if "NIH HHS" in i:
                found.append(i.split()[0] )
            else:
                found.append(i)
        tpc_found = []
        for i,found_list in zip(tpcname,Agency):
            for j in found_list:
                if "NIH HHS" in j:
                    tpc_found.append((i,j.split()[0]))
                else:
                    tpc_found.append((i,j))

    else:
        tpc_found = [   (i,j) for i,found_list in zip(tpcname,Agency) for j in found_list ]
    C_found__ = list(sorted(Counter(found).items(),key=lambda t: -t[1]))
    C_found = OrderedDict(C_found__)
    found =  [ i[0] for i in C_found__ ][:jou_num]

    C_tpc_found = Counter(tpc_found)
    num = list()
    for j in found:
        t_templist = list()
        for t in C_tpc.keys():
            t_templist.append(C_tpc_found[(t,j)])
        num.append(t_templist)
    tpc = C_tpc.keys()
    found = ['NCI', 'NINDS', 'NIGMS', 'NCRR', 'NCATS', 'NHLBI', 'NIBIB', 'Intramural NIH', 'NIDDK', 'NIAID', 'PHS HHS', 'Canadian Inst Hlth Res', 'NIEHS', 'NIH', 'Cancer Res UK', 'UK Med Res Council', 'Howard Hughes Med Inst', 'NIA', 'NICHD', 'NIMH', 'Wellcome Trust', 'NHGRI', 'NLM', 'NEI', 'National Natural Science Foundation of China', 'NIAMS', 'NIDA', 'National Institutes of Health', 'NIMHD', 'NIAAA', 'NIDCR', 'BLRD VA', 'Department of Health', 'National Cancer Institute', 'Biotechnology and Biological Sciences Research Council', 'FIC', 'CCR', 'NCCDPHP CDC HHS', 'CIHR', 'European Research Council', 'Worldwide Cancer Research', 'National Natural Science Foundation of China (National Science Foundation of China)', 'National Institute of Neurological Disorders and Stroke', 'Austrian Science Fund FWF', 'Telethon', 'Ministry of Science and Technology, Taiwan', 'FDA HHS', 'CSRD VA', 'NCCIH', 'National Research Foundation of Korea']
    found =['NCI', 'NINDS', 'NIGMS', 'NCRR', 'NCATS', 'NHLBI', 'NIBIB', 'PHS HHS', 'NIDDK', 'Intramural NIH', 'NIAID', 'Canadian Inst Hlth Res', 'Howard Hughes Med Inst', 'NIEHS', 'Cancer Res UK', 'UK Med Res Council', 'NIH', 'NICHD', 'NIA', 'NIMH']
    print(len(tpc))
    print(len(raw_name))
    num = np.array(num[:jou_num])
   
    im,bar = heatmap(num, found, tpc,
                    cmap="PuBu", cbarlabel="paper num ",vmin=0, vmax=200)
    texts = annotate_heatmap(im, valfmt="{x}",threshold=100)
    plt.show()

if __name__ == "__main__":
    createheadmap(20)
