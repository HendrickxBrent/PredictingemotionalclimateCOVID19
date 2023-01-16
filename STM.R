##### topic model

data <- rename(anger4STM2sdSeries)

library(stm)



### preprocessing

processed <- textProcessor(anger4STM2sdSeries$Tweet, metadata = anger4STM2sdSeries)

out <- prepDocuments(processed$documents, processed$vocab, processed$meta)
docs <- out$documents
vocab <- out$vocab
meta <- out$meta

# plotRemoved(processed$documents, lower.thresh = seq(1, 200, by = 100))
# out <- prepDocuments(processed$documents, processed$vocab, processed$meta, lower.thresh = 15)

# saveRDS(out, file = "STMAnger.rds")
# saveRDS(processed, file = "STM2Anger.rds")


poliblogPrevFit <- stm(documents = out$documents, vocab = out$vocab, K = 20, prevalence = ~series, max.em.its = 75, data = out$meta, init.type = "Spectral")

poliblogPrevFit10topics <- stm(documents = out$documents, vocab = out$vocab, K = 10, prevalence = ~series, max.em.its = 40, data = out$meta, init.type = "Spectral")


poliblogSelect <- selectModel(out$documents, out$vocab, K = 20, prevalence = ~emo_anger, max.em.its = 75, data = out$meta, runs = 20, seed = 8458159)

labelTopics(poliblogPrevFit, c(6, 13, 18))



plot(poliblogPrevFit, type = "summary", xlim = c(0, 0.3))

cloud(poliblogPrevFit, topic = 12, scale = c(4, 1)) # politics
cloud(poliblogPrevFit, topic = 19, scale = c(5, 1))
cloud(poliblogPrevFit, topic = 9, scale = c(5, 1)) # things going on in the world & pandemic

cloud(poliblogPrevFit, topic = 1, scale = c(5, 1)) # random
cloud(poliblogPrevFit, topic = 8, scale = c(5, 1)) # random
cloud(poliblogPrevFit, topic = 1, scale = c(5, 1)) # random
cloud(poliblogPrevFit, topic = 16, scale = c(5, 1)) # coronavirus, nhs, selfish, food, irrespons,distance, social, isol, staff, ignor, home
cloud(poliblogPrevFit, topic = 11, scale = c(5, 1)) # random verbs
cloud(poliblogPrevFit, topic = 3, scale = c(5, 1)) # social media & conversation
cloud(poliblogPrevFit, topic = 5, scale = c(5, 1)) # leisure time & weather(possibly because walking is leisure)
cloud(poliblogPrevFit, topic = 14, scale = c(5, 1))  #random
cloud(poliblogPrevFit, topic = 4, scale = c(5, 1))  #some great British mockery
cloud(poliblogPrevFit, topic = 18, scale = c(5, 1))  #random
cloud(poliblogPrevFit, topic = 20, scale = c(5, 1))  #random
cloud(poliblogPrevFit, topic = 10, scale = c(5, 1))  #football
cloud(poliblogPrevFit, topic = 7, scale = c(5, 1))  #random
cloud(poliblogPrevFit, topic = 15, scale = c(5, 1))  #random
cloud(poliblogPrevFit, topic = 13, scale = c(5, 1))  # school, children, student, teacher, lesson, educ, video, 
cloud(poliblogPrevFit, topic = 1, scale = c(5, 1))  #random
cloud(poliblogPrevFit, topic = 17, scale = c(5, 1))  #politics countries of the UK (Scotland, northen Ireland,...)


mod.out.corr <- topicCorr(poliblogPrevFit)
plot(mod.out.corr)

## now with 10 topics

library(parallel)
ncores <- 10

cl <- makeCluster(ncores)
# Fit the STM model in parallel
fit <- parLapply(cl, 1:ncores, function(i) {
  stm(documents = out$documents, vocab = out$vocab, K = 10, prevalence = ~series, max.em.its = 75, data = out$meta, init.type = "Spectral")
})

# Stop the parallel processing environment
stopCluster(cl)
poliblogPrevFit2 <- searchK(documents = out$documents, vocab = out$vocab, K = c(5, 10,15), prevalence = ~series, max.em.its = 50, data = out$meta, init.type = "Spectral", cores=10)









poliblogPrevFit10topics <- stm(documents = out$documents, vocab = out$vocab, K = 10, prevalence = ~series, max.em.its = 40, data = out$meta, init.type = "Spectral")
plot(poliblogPrevFit10topics, type = "summary", xlim = c(0, 0.3))
cloud(poliblogPrevFit10topics, topic = 3, scale = c(4, 1))
cloud(poliblogPrevFit10topics, topic = 7, scale = c(4, 1))
cloud(poliblogPrevFit10topics, topic = 9, scale = c(4, 1))
cloud(poliblogPrevFit10topics, topic = 8, scale = c(4, 1))
cloud(poliblogPrevFit10topics, topic = 1, scale = c(4, 1))
cloud(poliblogPrevFit10topics, topic = 6, scale = c(4, 1))
cloud(poliblogPrevFit10topics, topic = 4, scale = c(4, 1))
cloud(poliblogPrevFit10topics, topic = 5, scale = c(4, 1))




save.image(file="Anger.RData")






### fear


plot(poliblogPrevFit, type = "summary", xlim = c(0, 0.3))
cloud(poliblogPrevFit, topic = 18, scale = c(4, 1)) # anxiety related terms
cloud(poliblogPrevFit, topic = 12, scale = c(4, 1)) # anxiety related terms
cloud(poliblogPrevFit, topic = 13, scale = c(4, 1)) # anxiety related terms
cloud(poliblogPrevFit, topic = 15, scale = c(4, 1)) # coronavirus, covid, death, symptom, hospit, case, contract, pandem,test,
cloud(poliblogPrevFit, topic = 11, scale = c(4, 1)) # 
cloud(poliblogPrevFit, topic = 19, scale = c(4, 1)) # also restrictions a bit
cloud(poliblogPrevFit, topic = 2, scale = c(4, 1))
cloud(poliblogPrevFit, topic = 10, scale = c(4, 1))

cloud(poliblogPrevFit, topic = 10, scale = c(4, 1))
cloud(poliblogPrevFit, topic = 17, scale = c(4, 1))
cloud(poliblogPrevFit, topic = 14, scale = c(4, 1))
cloud(poliblogPrevFit, topic = 1, scale = c(4, 1))
cloud(poliblogPrevFit, topic = 8, scale = c(4, 1))
cloud(poliblogPrevFit, topic = 5, scale = c(4, 1))
cloud(poliblogPrevFit, topic = 4, scale = c(4, 1))

mod.out.corr <- topicCorr(poliblogPrevFit)
plot(mod.out.corr)
View(mod.out.corr)











