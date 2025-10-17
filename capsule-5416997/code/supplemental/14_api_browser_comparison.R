cat(rep('=', 80),
    '\n\n',
    'OUTPUT FROM: supplemental/14_api_browser_comparison.R',
    '\n\n',
    sep = ''
    )

# Load the API tree and the natural tree datasets

api.tree <- read.csv('../data/supplemental/api_tree.csv')
natural.tree <- readRDS('../data/supplemental/natural_tree.rds')

#############
# Figure S3 #
#############

natural.tree$w <- NA
natural.tree$w[natural.tree$step == 1] <- 1 # All recs in the first step come from a common origin video
natural.tree$w[natural.tree$step == 2] <- 1 # All recs in the second step come from a unique origin (one of the 20 recs from the first video)

# If a recommendation appears in a step multiple times, we only get recs for that video once, but upweight those videos according to freq
for(i in 1:nrow(natural.tree)){
    if(natural.tree$step[i] == 1) next
    if(natural.tree$step[i] == 2) next
    paths.to.rec <- (natural.tree$originID[i] == natural.tree$recID) & (natural.tree$step == natural.tree$step[i] - 1)
    sum(natural.tree$w[paths.to.rec])    
    natural.tree$w[i] <- sum(natural.tree$w[paths.to.rec])
}

weighted.means <- c(weighted.mean(natural.tree$in.api.tree[natural.tree$step == 1], w = natural.tree$w[natural.tree$step == 1]),
                    weighted.mean(natural.tree$in.api.tree[natural.tree$step == 2], w = natural.tree$w[natural.tree$step == 2]),
                    weighted.mean(natural.tree$in.api.tree[natural.tree$step == 3], w = natural.tree$w[natural.tree$step == 3]),
                    weighted.mean(natural.tree$in.api.tree[natural.tree$step == 4], w = natural.tree$w[natural.tree$step == 4]),
                    weighted.mean(natural.tree$in.api.tree[natural.tree$step == 5], w = natural.tree$w[natural.tree$step == 5])
                    )

pdf('../results/proportion_by_step_in_tree_weighted.pdf')
barplot(weighted.means,
        main = 'Weighted Proportion of Natural Recs in API Tree (In Step)',
        names.arg = c('Step 1', 'Step 2', 'Step 3', 'Step 4', 'Step 5'),
        ylab = 'Proportion of Naturalistic Recs in API Tree',
        ylim = c(0,1)
        )
dev.off()

###############################
# Table in Appendix Section 4 #
###############################

set.seed(63110)
to.label <- natural.tree[sample(which(natural.tree$in.api.tree == 0 & natural.tree$step == 5), 10),]
step.five.recs <- to.label$recID

for(i in 1:nrow(to.label)){    
    step.six.origin <- to.label$recID[i]
    step.six.origin.str <- paste0('\\verb|', step.six.origin, '|')
    if(!(step.six.origin %in% api.tree$RecID)) step.six.origin.str <- paste0(step.six.origin.str, '***')
        
    step.five.origin <- to.label$originID[i]
    step.five.origin.str <- paste0('\\verb|', step.five.origin, '|')
    if(!(step.five.origin %in% api.tree$RecID)) step.five.origin.str <- paste0(step.five.origin.str, '***')

    step.four.origin <- natural.tree$originID[natural.tree$step == 4 & natural.tree$recID == step.five.origin][1]
    step.four.origin.str <- paste0('\\verb|', step.four.origin, '|')
    if(!(step.four.origin %in% api.tree$RecID)) step.four.origin.str <- paste0(step.four.origin.str, '***')
    
    step.three.origin <- natural.tree$originID[natural.tree$step == 3 & natural.tree$recID == step.four.origin][1]
    step.three.origin.str <- paste0('\\verb|', step.three.origin, '|')
    if(!(step.three.origin %in% api.tree$RecID)) step.three.origin.str <- paste0(step.three.origin.str, '***')

    step.two.origin <- natural.tree$originID[natural.tree$step == 2 & natural.tree$recID == step.three.origin][1]
    step.two.origin.str <- paste0('\\verb|', step.two.origin, '|')
    if(!(step.two.origin %in% api.tree$RecID)) step.two.origin.str <- paste0(step.two.origin.str, '***')

    step.one.origin <- natural.tree$originID[natural.tree$step == 1 & natural.tree$recID == step.two.origin][1]
    step.one.origin.str <- paste0('\\verb|', step.one.origin, '|')
    if(!(step.one.origin %in% api.tree$RecID)) step.one.origin.str <- paste0(step.one.origin.str, '***')

    row <- paste(step.six.origin.str,
                 step.five.origin.str,
                 step.four.origin.str,
                 step.three.origin.str,
                 step.two.origin.str,
                 step.one.origin.str,
                 sep = ' & ')
    row <- paste0(row, ' \\\\')
    
    cat(row)
    cat('\n')
}
