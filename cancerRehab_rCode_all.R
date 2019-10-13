setwd(".")
Sys.setlocale(category = "LC_ALL",locale = "English_United States.1252")
source("pubmedXML.R")
theData <- extract_xml("you_xml_from_pubmed.xml")
write.csv(theData, file = "pubmed2csv.csv", row.names = FALSE)
years <- sapply(split(theData, theData$year), nrow)
write.csv(years, file = "pubsPerYear.csv")
library(plyr)
ptypes <- count(unlist(strsplit(theData$ptype, "|", fixed = TRUE)))
ptypes <- ptypes[order(ptypes$freq),]
write.csv(ptypes, file = "pubsPerPubType.csv")
mesh <- count(unlist(strsplit(theData$meshHeadings, "|", fixed = TRUE)))
mesh <- mesh[order(mesh$freq),]
write.csv(mesh, file = "pubsPerMeshHeading.csv", row.names = FALSE)
funders <- count(unlist(strsplit(theData$grantAgency, "|", fixed = TRUE)))
funders <- funders[order(funders$freq),]
write.csv(funders, file = "pubsPerFunder.csv", row.names = FALSE)
tail(funders, 20)
meshYrs <- theData[, c(5, 12)]
library(splitstackshape)
library(reshape2)
meshYrs <- cSplit(meshYrs, "meshHeadings", "|", direction = "long")
meshYrs <- dcast(meshYrs, meshHeadings ~ year)
meshYrs$total <- rowSums(meshYrs[2:25])
meshYrs$percChange <- (meshYrs[,26] - meshYrs[,2]) / meshYrs[,2]
write.csv(meshYrs, file = "pubsPerMeshPerYr.csv", row.names = FALSE)
journals <- sapply(split(theData, theData$journal), nrow)
journals <- sort(journals)
write.csv(journals, file = "pubsPerJournal.csv")
tail(journals, 10)
fcntry <- count(unlist(strsplit(theData$grantCountry, "|", fixed = TRUE)))
fcntry <- fcntry[order(fcntry$freq),]
fcntry
write.csv(fcntry, file = "pubsPerGrantCountry.csv", row.names = FALSE)


theDF <- read.csv("cancerByTherapy.csv")
row.names(theDF) <- theDF$Term
theMtx <- as.matrix(theDF[,2:13])
library(RColorBrewer)
heatmap1 <- heatmap(theMtx, Rowv = NA, Colv = NA, col = brewer.pal(11, "RdYlBu"), scale = "none")


library(tm)
library(slam)
theData <- read.csv("pubmed_result_parsed_without_NAabstract.csv", stringsAsFactors = FALSE)
myStopwords <- scan("stopwords.txt", what = "varchar", skip = 1)
docs <- data.frame(doc_id = theData$pmid, text = theData$abstract)
pmids <- as.vector(theData$pmid)
corpDocs <- Corpus(DataframeSource(docs))
corpDocs <- tm_map(corpDocs, removePunctuation)
corpDocs <- tm_map(corpDocs, content_transformer(tolower))
corpDocs <- tm_map(corpDocs, removeWords, stopwords("English"))
corpDocs <- tm_map(corpDocs, removeWords, myStopwords)
corpDocs <- tm_map(corpDocs, stemDocument)
corpDocs <- tm_map(corpDocs, stripWhitespace)
dtm <- DocumentTermMatrix(corpDocs)
rownames(dtm) <- pmids
dtm <- dtm[row_sums(dtm) > 0,]
dtm
dtm <- removeSparseTerms(dtm, 0.995)
dtm
library(topicmodels)
seed <- list(1379, 6513, 10719, 16007, 20991)
ldaOut <- LDA(dtm, 50, method = "Gibbs", control = list(nstart = 5, seed = seed, best = TRUE, burnin = 4000, iter = 2000, thin = 500))
primaryTopics <- as.matrix(topics(ldaOut))
write.csv(primaryTopics, file = "docsToTopics.csv")
topicTerms <- as.matrix(terms(ldaOut, 15))
write.csv(topicTerms, file = "topicsToTerms.csv")
topicProbs <- as.data.frame(ldaOut@gamma)
write.csv(topicProbs, file = "topicProbs.csv")
topicLists <- topics(ldaOut, threshold = 0.08)
topicLists <- sapply(topicLists, paste0, collapse = "|")
newData <- merge(theData, primaryTopics, by.x = "pmid", by.y = "row.names", all.x = TRUE)
newData <- merge(newData, topicLists, by.x = "pmid", by.y = "row.names", all.x = TRUE)
write.csv(newData, file = "clusteredData.csv", row.names = FALSE)
library(igraph)
edges <- as.matrix(topics(ldaOut, 2))
edges <- as.data.frame(t(edges))
nCounts <- sapply(split(edges, edges$V1), nrow)
nodes <- data.frame(name=names(nCounts), nCounts)

theGraph <- graph_from_data_frame(edges, directed = FALSE, vertices = nodes)
write_graph(theGraph, file = "topicNetwork.G.", format = "graphml")
write.csv(nodes, file = "nodeList.csv")
library(plyr)
library(reshape2)
tyears <- count(newData, vars = c("V1", "year"))
tyears <- dcast(tyears, V1 ~ year, value.var = "freq")
head(tyears)
tyears$total <- rowSums(tyears[,2:29], na.rm = TRUE)
tyears$percChng <- (tyears[,27] - tyears[,3]) / tyears[,3]
write.csv(tyears, file = "pubsPerTopicPerYear.csv", row.names = FALSE)
library(splitstackshape)
tfund <- newData[,c(12, 16)]
tfund <- cSplit(tfund, "grantAgency", "|", direction = "long")
tfund <- dcast(tfund, grantAgency ~ V1)
head(tfund)
tfund$total <- rowSums(tfund[,2:52])
tfund <- tfund[order(tfund$total),]
tail(tfund, 15)
write.csv(tfund, file = "pubsPerTopicPerFunder.csv", row.names = FALSE)