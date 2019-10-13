from lxml import etree
f = open("you_xml_from_pubmed.xml",encoding="utf8")
print("a")
txt = f.read()
print("aa")
html = etree.HTML(txt)
print("aaa")
tree = etree.ElementTree(html)
#pickle.dump(tree,open("newdata\Glioma.pkl","wb"))
print("bbb")
PubMedPubDate_year = []
pubdate = tree.xpath('//journal/journalissue/pubdate')
pmid = tree.xpath('//pubmedarticle/medlinecitation/pmid')
print(len(pubdate),len(pmid))
for h,idx in zip(pubdate,pmid):
    if h.xpath("./year"):
        PubMedPubDate_year.append(int(h.xpath("./year")[0].text))
    elif h.xpath("./medlinedate"):
        year = h.xpath("./medlinedate")[0].text
        try:
            year = int(year[:4])
        except :
            year = int(year[-4:])
        PubMedPubDate_year.append(year)
    else:
        print("can't find year")
        print(idx.text)
        PubMedPubDate_year.append("")
del tree,html,txt
print(len(PubMedPubDate_year))
print(max(PubMedPubDate_year),min(PubMedPubDate_year))

import pandas as pd
df = pd.read_csv("youcsv_produce_by_pubmedXML.csv",sep=',',encoding="utf8")
df.year = PubMedPubDate_year
df.to_csv("CSV_YearModified.csv")