graphics.off()
rm(list=ls(all=TRUE))
require(parallel)
require(rstan)
require(shinystan)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

cmrc <- stanc(file='cmr_fr_revised.stan', model_name='cmr')
cmr <- stan_model(stanc_ret = cmrc, model_name='cmr')

nSubjTries <- 100

load('mo1970_data.rdata')

nSubj <- max(subject)
LL <- rep(20, length(subject))
maxLR <- ncol(recall)
nSeq <- nrow(recall)

datalist <- list(
    LL=LL, maxLR=maxLR, nSeq=nSeq, nSubj=nSubj,
    subj=subject,
    recall_sequences=recall
)

subjPars <- c()

for (s in 1:nSubj) {
    subjRecall <- recall[subject == s,]

    subjData <- list(
        LL=rep(20, nrow(subjRecall)), maxLR=maxLR, nSeq=nrow(subjRecall), nSubj=1,
        subj=rep(1, nrow(subjRecall)),
        recall_sequences=subjRecall
    )

    subjMAPFit <- optimizing(cmr, data = subjData, init = '0', verbose = TRUE, as_vector = FALSE, iter=10000)

    for (i in 1:nSubjTries) {
        if (subjMAPFit$return_code != 0) {
            subjMAPFit <- optimizing(cmr, data = subjData, init = 'random', verbose = TRUE, as_vector = FALSE, iter=10000)
        } else {
            break
        }
    }

    if (subjMAPFit$return_code == 0) {
        subjPars <- rbind(subjPars, subjMAPFit$par$parMean)
    } else {
        subjPars <- rbind(subjPars, rep(0, ncol(subjPars)))
    }
}

zSubj <- scale(subjPars, scale=TRUE, center=TRUE)

subjInit <- list(
    parMean = attr(zSubj, 'scaled:center'),
    subjCorrChol = chol(cor(zSubj)),
    subjSD = attr(zSubj, 'scaled:scale'),
    zSubj = t(zSubj[1:(nSubj - 1),])
)

adaptSteps <- 1000
nChains <- 5
thinSteps <- 1
numSteps <- 1000

initlist <- list()

for (i in 1:nChains) {
    initlist[[i]] <- subjInit
}

fit <- sampling(cmr, pars=c('subjPars', 'pred_recall'), data=datalist, init=initlist, chains=nChains, iter=numSteps + adaptSteps, warmup=adaptSteps, verbose=TRUE, control=list(adapt_delta=0.9))

fitSummary <- summary(fit, 'subjPars')

subjPars <- extract(fit, pars='subjPars')$subjPars
pred <- extract(fit, pars='pred_recall')$pred_recall
predCollected <- matrix(0, nrow=prod(dim(pred)[1:2]), ncol=dim(pred)[3])
for (i in 1:dim(pred)[1]) {
    predCollected[(i - 1) * dim(pred)[2] + 1:dim(pred)[2],] <- pred[i,,]
}

subjParMean <- apply(subjPars, MARGIN=c(1, 2), FUN=mean)
subjParSD <- apply(subjPars, MARGIN=c(1, 2), FUN=sd)

parNames <- c('init_fc', 'beta_enc', 'beta_rec', 'beta_start', 'P1', 'P2', 'stopInit', 'stopScale', 'stopShape', 'gamma')

subjParDF <- c()

for (i in 1:length(parNames)) {
    subjParDF <- rbind(subjParDF,
        data.frame(par=parNames[i], subject=rep(1:dim(subjPars)[3], each=dim(subjPars)[1]), y=c(subjPars[,i,]))
    )
}

subjParDF <- rbind(subjParDF,
    data.frame(par='Mean correct', subject=1:nSubj, y=tapply(apply(recall, MARGIN=1, FUN=function(x) sum(x > 0)) / 20, INDEX=subject, FUN=mean))
)

rm(subjPars)

x11()
print(ggplot(data=data.frame(sample=c(subjParMean), parameter=factor(rep(parNames, each=nrow(subjParMean)), levels=parNames)), mapping=aes(x=sample)) + geom_histogram(bins=30) + facet_wrap('parameter', scales='free'))

x11()
print(ggplot(data=subjParDF, mapping=aes(x=subject, y=y)) + stat_summary(fun.data='median_hilow') + geom_hline(data=data.frame(par=parNames, yintercept=apply(subjParMean, MARGIN=2, FUN=median)), mapping=aes(yintercept=yintercept), linetype='dashed', color='red') + geom_hline(data=data.frame(par=parNames, yintercept=apply(subjParMean, MARGIN=2, FUN=quantile, probs=0.025)), mapping=aes(yintercept=yintercept), linetype='dotted', color='red') + geom_hline(data=data.frame(par=parNames, yintercept=apply(subjParMean, MARGIN=2, FUN=quantile, probs=0.975)), mapping=aes(yintercept=yintercept), linetype='dotted', color='red') + facet_grid(~ par, scales='free_x') + coord_flip())

source('plot_functions.r')

x11()
print(plot.serial.position(obs_recall=recall, obs_subj=subject, pred_recall=predCollected))

x11()
print(plot.lag.crp(obs_recall=recall, pred_recall=predCollected))

x11()
print(plot.stop.prob(obs_recall=recall, pred_recall=predCollected))
