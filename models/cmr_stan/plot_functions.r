se <- function(x, na.rm=TRUE) {
    y <- x[!is.na(x)]

    return(sd(y) / sqrt(length(y)))
}

plot.serial.position <- function(obs_recall, obs_subj = rep(1, nrow(obs_recall)), pred_recall = NULL, pred_subj = NULL) {
    require(ggplot2)

    maxPos <- max(obs_recall, na.rm=TRUE)

    obsCounts <- apply(obs_recall, MARGIN=1, FUN=function(x) {tabulate(x[x > 0], nbins=maxPos)})
    if (!is.null(obs_subj)) {
        rawCounts <- obsCounts

        obsCounts <- c()
        for (s in unique(obs_subj)) {
            obsCounts <- cbind(obsCounts, rowMeans(rawCounts[,obs_subj==s]))
        }
    }

    plotData <- data.frame(pos=1:maxPos, mean=rowMeans(obsCounts), se=apply(obsCounts, MARGIN=1, FUN=se))

    if (is.null(pred_recall)) {
        return(ggplot(data=plotData, mapping=aes(x=pos, y=mean, ymin=mean-se, ymax=mean+se)) + geom_line() + geom_pointrange())
    } else {
        predCounts <- apply(pred_recall, MARGIN=1, FUN=function(x) {tabulate(x[x > 0], nbins=maxPos)})

        plotData <- rbind(
            cbind(source='Observed', plotData),
            data.frame(source='Predicted', pos=1:maxPos, mean=rowMeans(predCounts), se=apply(predCounts, MARGIN=1, FUN=se))
        )

        return(ggplot(data=plotData, mapping=aes(x=pos, y=mean, ymin=mean-se, ymax=mean+se, color=source, linetype=source)) + geom_line() + geom_pointrange())
    }
}

plot.lag.crp <- function(obs_recall, pred_recall = NULL) {
    require(ggplot2)

    maxPos <- max(obs_recall, na.rm=TRUE)
    lagLevels <- seq(-maxPos, maxPos)
    nLevels <- length(lagLevels)

    obsLag <- apply(obs_recall, MARGIN=1, FUN=diff)

    obsCounts <- apply(obsLag, MARGIN=2, FUN=function(x) {tabulate(factor(x, levels=lagLevels), nbins=nLevels)})
    obsCounts[lagLevels==0,] <- 0
    obsP <- obsCounts / matrix(colSums(obsCounts), nrow=nrow(obsCounts), ncol=ncol(obsCounts), byrow=TRUE)
    obsP[is.na(obsP)] <- 0
    obsP[lagLevels==0,] <- NA

    plotData <- data.frame(lag=lagLevels, mean=rowMeans(obsP), se=apply(obsP, MARGIN=1, FUN=se))

    if (is.null(pred_recall)) {
        return(ggplot(data=plotData, mapping=aes(x=lag, y=mean, ymin=mean-se, ymax=mean+se)) + geom_line() + geom_pointrange())
    } else {
        predLag <- apply(pred_recall, MARGIN=1, FUN=diff)

        predCounts <- apply(predLag, MARGIN=2, FUN=function(x) {tabulate(factor(x, levels=lagLevels), nbins=nLevels)})
        predCounts[lagLevels==0,] <- 0
        predP <- predCounts / matrix(colSums(predCounts), nrow=nrow(predCounts), ncol=ncol(predCounts), byrow=TRUE)
        predP[is.na(predP)] <- 0
        predP[lagLevels==0,] <- NA

        plotData <- rbind(
            cbind(source='Observed', plotData),
            data.frame(source='Predicted', lag=lagLevels, mean=rowMeans(predP), se=apply(predP, MARGIN=1, FUN=se))
        )

        return(ggplot(data=plotData, mapping=aes(x=lag, y=mean, ymin=mean-se, ymax=mean+se, color=source, linetype=source)) + geom_line() + geom_pointrange())
    }
}

plot.stop.prob <- function(obs_recall, pred_recall = NULL) {
    require(ggplot2)

    obsStop <- 1 - (obs_recall != 0)

    plotData <- data.frame(pos=1:ncol(obsStop), mean=colMeans(obsStop), se=apply(obsStop, MARGIN=2, FUN=se))

    if (is.null(pred_recall)) {
        return(ggplot(data=plotData, mapping=aes(x=pos, y=mean, ymin=mean-se, ymax=mean+se)) + geom_line() + geom_pointrange())
    } else {
        predStop <- 1 - (pred_recall != 0)

        plotData <- rbind(
            cbind(source='Observed', plotData),
            data.frame(source='Predicted', pos=1:ncol(predStop), mean=colMeans(predStop), se=apply(predStop, MARGIN=2, FUN=se))
        )

        return(ggplot(data=plotData, mapping=aes(x=pos, y=mean, ymin=mean-se, ymax=mean+se, color=source, linetype=source)) + geom_line() + geom_pointrange())
    }
}
# 
# plot.pfr <- function(obs_recall, obs_subj = rep(1, nrow(obs_recall)), pred_recall = NULL, pred_subj = NULL) {
#     require(ggplot2)
#
#     maxPos <- max(obs_recall, na.rm=TRUE)
#
#     obsCounts <- apply(obs_recall[,1], MARGIN=1, FUN=function(x) {tabulate(x[x > 0], nbins=maxPos)})
#     if (!is.null(obs_subj)) {
#         rawCounts <- obsCounts
#
#         obsCounts <- c()
#         for (s in unique(obs_subj)) {
#             obsCounts <- cbind(obsCounts, rowMeans(rawCounts[,obs_subj==s]))
#         }
#     }
#
#     plotData <- data.frame(pos=1:maxPos, mean=rowMeans(obsCounts), se=apply(obsCounts, MARGIN=1, FUN=se))
#
#     if (is.null(pred_recall)) {
#         return(ggplot(data=plotData, mapping=aes(x=pos, y=mean, ymin=mean-se, ymax=mean+se)) + geom_line() + geom_pointrange())
#     } else {
#         predCounts <- apply(pred_recall, MARGIN=1, FUN=function(x) {tabulate(x[x > 0], nbins=maxPos)})
#
#         plotData <- rbind(
#             cbind(source='Observed', plotData),
#             data.frame(source='Predicted', pos=1:maxPos, mean=rowMeans(predCounts), se=apply(predCounts, MARGIN=1, FUN=se))
#         )
#
#         return(ggplot(data=plotData, mapping=aes(x=pos, y=mean, ymin=mean-se, ymax=mean+se, color=source, linetype=source)) + geom_line() + geom_pointrange())
#     }
# }
