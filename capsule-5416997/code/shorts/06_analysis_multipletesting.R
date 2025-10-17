cat(rep('=', 80),
    '\n\n',
    'OUTPUT FROM: shorts/06_analysis_multipletesting.R',
    '\n\n',
    sep = ''
    )
    
library(data.table)
library(car)
library(sandwich)
library(lmtest)
library(ggplot2)
library(tidyverse)

###############
## functions ##
###############

`%.%` <- paste0

simes <- function(ps){
  min(sort(length(ps) * ps / rank(ps)))
}

### functions to handle inconsistent interaction ordering of mlm() ###

## convert interaction terms of form 'b#:a#' to 'a#:b#'
reorder.interaction.names <- function(x, prefix = ''){
  x <- gsub('^' %.% prefix, '', x)
  sapply(strsplit(x, ':'),
         function(y){
           paste(sort(y), collapse = ':')
         })
}

## take term of form 'a1:b1', look up in vector of form 'b#:a#, return 'b1:a1'
convert.interaction.names <- function(x, y, prefix.y = ''){
  ind <- match(reorder.interaction.names(x),
               reorder.interaction.names(y, prefix = prefix.y)
  )
  return(y[ind])
}

## modified from print.linearHypothesis.mlm to use alternate df & return pvals
##   (print method is responsible for doing the actual computation of pvals)
extract.lht <- function(x,
                        SSP = TRUE,
                        SSPE = SSP,
                        digits = getOption('digits'),
                        df.residual = x$df.residual
){
  test <- x$test
  if (!is.null(x$P) && SSP) {
    P <- x$P
    cat("\n Response transformation matrix:\n")
    attr(P, "assign") <- NULL
    attr(P, "contrasts") <- NULL
    print(P, digits = digits)
  }
  if (SSP) {
    cat("\nSum of squares and products for the hypothesis:\n")
    print(x$SSPH, digits = digits)
  }
  if (SSPE) {
    cat("\nSum of squares and products for error:\n")
    print(x$SSPE, digits = digits)
  }
  if ((!is.null(x$singular)) && x$singular) {
    warning("the error SSP matrix is singular; multivariate tests are unavailable")
    return(invisible(x))
  }
  SSPE.qr <- qr(x$SSPE)
  eigs <- Re(eigen(qr.coef(SSPE.qr, x$SSPH), symmetric = FALSE)$values)
  tests <- matrix(NA, 4, 4)
  rownames(tests) <- c("Pillai", "Wilks", "Hotelling-Lawley",
                       "Roy")
  if ("Pillai" %in% test)
    tests[1, 1:4] <- car:::Pillai(eigs, x$df, df.residual)
  if ("Wilks" %in% test)
    tests[2, 1:4] <- car:::Wilks(eigs, x$df, df.residual)
  if ("Hotelling-Lawley" %in% test)
    tests[3, 1:4] <- car:::HL(eigs, x$df, df.residual)
  if ("Roy" %in% test)
    tests[4, 1:4] <- car:::Roy(eigs, x$df, df.residual)
  tests <- na.omit(tests)
  ok <- tests[, 2] >= 0 & tests[, 3] > 0 & tests[, 4] > 0
  ok <- !is.na(ok) & ok
  tests <- cbind(x$df, tests, pf(tests[ok, 2], tests[ok, 3],
                                 tests[ok, 4], lower.tail = FALSE))
  colnames(tests) <- c("Df", "test stat", "approx F", "num Df",
                       "den Df", "Pr(>F)")
  tests <- structure(as.data.frame(tests),
                     heading = paste("\nMultivariate Test",
                                     if (nrow(tests) > 1)
                                       "s", ": ", x$title, sep = ""),
                     class = c("anova",
                               "data.frame"
                     )
  )
  return(tests)
}

###############
## load data ##
###############

d <- fread('../results/intermediate data/shorts/qualtrics_w12_clean_ytrecs_may2024.csv')

##############
## controls ##
##############

platform.controls <- c('age_cat',
                       'male',
                       'pol_interest',
                       'freq_youtube')

mwpolicy.controls <- 'mw_index_pre'

media.controls <- c('trust_majornews',
                    'trust_youtube',
                    'fabricate_majornews',
                    'fabricate_youtube')

affpol.controls <- c('affpol_smart',
                     'affpol_comfort')

controls.raw <- unique(c(platform.controls,
                         mwpolicy.controls,
                         media.controls,
                         affpol.controls))

## transform control variables by creating dummies and demeaning
controls.trans <- list()
for (j in controls.raw){
  ## convert to dummies if needed
  controls.j <- model.matrix(as.formula('~ 0 + ' %.% j),
                             model.frame(as.formula('~ 0 + ' %.% j),
                                         data = d,
                                         na.action = 'na.pass'
                             )
  )
  ## demean by column
  controls.j <- sweep(controls.j,
                      MARGIN = 2,
                      STATS = colMeans(controls.j, na.rm = TRUE),
                      FUN = `-`,
  )
  colnames(controls.j) <- make.names(colnames(controls.j))
  ## remove control from original data
  d[[j]] <- NULL
  ## reinsert transformed control
  d <- cbind(d, controls.j)
  ## keep track of which original controls map to which transformed controls
  controls.trans[[j]] <- colnames(controls.j)
}

## map original control variables to transformed versions
platform.controls <- unlist(controls.trans[platform.controls])
mwpolicy.controls <- unlist(controls.trans[mwpolicy.controls])
media.controls <- unlist(controls.trans[media.controls])
affpol.controls <- unlist(controls.trans[affpol.controls])

### Platform interactions ###
d <- d %>% filter(!is.na(interface_duration)) # -- 929 observations

##############
## outcomes ##
##############

### HYPOTHESIS FAMILY: MIN WAGE POLICY ATTITUDES ###

## ONLY HAVE ONE OUTCOME
mwpolicy.outcomes <- 'mw_index'

outcomes <- unique(c(mwpolicy.outcomes))

################
## treatments ##
################

## CREATE ATTITUDE DUMMIES
# 1-LIBERALS, 2-MODERATES, 3-CONSERVATIVES
d[, attitude := c('pro', 'neutral', 'anti')[thirds]]
d[, attitude.pro := as.numeric(attitude == 'pro')]
d[, attitude.neutral := as.numeric(attitude == 'neutral')]
d[, attitude.anti := as.numeric(attitude == 'anti')]

## CREATE SEQUENCE DUMMIES -- AC, PC, AI, PI
d[, recsys.ac := as.numeric(treatment_arm %like% 'ac')]
d[, recsys.pc := as.numeric(treatment_arm %like% 'pc')]
d[, recsys.ai := as.numeric(treatment_arm %like% 'ai')]
d[, recsys.pi := as.numeric(treatment_arm %like% 'pi')]

# (a)  Increasing vs.  Constant assignment among Pro participants;
# (b)  Increasing vs.  Constant assignment among Anti participants;
# (c)  Increasing  vs.   Constant  assignment  among  Moderate  participants  assigned  to  a  Prosequence;
# (d)  Increasing vs.  Constant assignment among moderate participants assigned to an Antisequence;
# (e)  Pro vs.  Anti sequence assignment among moderate participants with Increasing assignment;
# (f)  Pro vs.  Anti seed among moderate participants with Constant assignment.

# Treatments:
treatments <- c('attitude.pro:recsys.pi', # (a)
                'attitude.pro:recsys.pc', # (a)
                'attitude.anti:recsys.ai', # (b)
                'attitude.anti:recsys.ac', # (b)
                'attitude.neutral:recsys.ai', # (d-e)
                'attitude.neutral:recsys.pi', # (c-e)
                'attitude.neutral:recsys.ac', # (d-f)
                'attitude.neutral:recsys.pc') # (c-f)

# Contrasts:
contrasts <- rbind(
  # Increasing vs. Constant assignment among Pro participants
  i = c(treat = 'attitude.pro:recsys.pi',
        ctrl = 'attitude.pro:recsys.pc'
  ),
  # Increasing vs. Constant assignment among Anti participants
  ii = c(treat = 'attitude.anti:recsys.ai',
         ctrl = 'attitude.anti:recsys.ac'
  ),
  # Increasing vs. Constant assignment among Moderate participants assigned to a Pro sequence
  iii = c(treat = 'attitude.neutral:recsys.pi',
          ctrl = 'attitude.neutral:recsys.pc'
  ),
  # Increasing vs. Constant assignment among moderate participants assigned to an Anti sequence
  iv = c(treat = 'attitude.neutral:recsys.ai',
         ctrl = 'attitude.neutral:recsys.ac'
  ),
  # Pro vs. Anti sequence assignment among moderate participants with Increasing assignment
  v = c(treat = 'attitude.neutral:recsys.ai',
        ctrl = 'attitude.neutral:recsys.pi'
  ),
  # Pro vs. Anti sequence assignment among moderate participants with Constant assignment
  vi = c(treat = 'attitude.neutral:recsys.ac',
         ctrl = 'attitude.neutral:recsys.pc'
  )
)

##########################
## hierarchical testing ##
##########################

## initialize top layer p-values:
## does treatment have any effect on any outcome in family
families <- c('mwpolicy')
layer1.pvals <- rep(NA_real_, length(families))
layer1.notes <- rep('', length(families))
names(layer1.pvals) <- families

## initialize 2nd layer p-values:
##   which treatment has detectable effect?
contrast.pvals <- rep(NA_real_, nrow(contrasts))
names(contrast.pvals) <- paste(contrasts[, 'treat'],
                               contrasts[, 'ctrl'],
                               sep = '.vs.'
)
layer2.pvals <- list( mwpolicy = contrast.pvals)
rm(contrast.pvals)

## initialize 3rd layer p-values:
##   on which specific outcome in family?
layer3.pvals <- list()
layer3.ests <- list()
layer3.ses <- list()
layer3.notes <- list()
for (i in 1:length(families)){
  family <- families[i]
  layer3.pvals[[family]] <- list()
  layer3.ests[[family]] <- list()
  layer3.ses[[family]] <- list()
  layer3.notes[[family]] <- list()
  outcomes <- get(family %.% '.outcomes')
  for (j in 1:nrow(contrasts)){
    contrast <- paste(contrasts[j, 'treat'],
                      contrasts[j, 'ctrl'],
                      sep = '.vs.'
    )
    layer3.pvals[[family]][[contrast]] <- numeric(0)
    layer3.ests[[family]][[contrast]] <- numeric(0)
    layer3.ses[[family]][[contrast]] <- numeric(0)
    for (k in 1:length(outcomes)){
      outcome <- outcomes[k]
      layer3.pvals[[family]][[contrast]][outcome] <- NA_real_
      layer3.ests[[family]][[contrast]][outcome] <- NA_real_
      layer3.ses[[family]][[contrast]][outcome] <- NA_real_
      layer3.notes[[family]][outcome] <- ''
    }
  }
}

### begin nested analyses ###
for (i in 1:length(families)){
  
  family <- families[i]
  family.outcomes <- get(family %.% '.outcomes')
  family.controls <- get(family %.% '.controls')

  
  family.controls.interactions <- as.character(
    outer(treatments,
          family.controls,
          FUN = function(x, y) x %.% ':' %.% y
    )
  )
  
  family.formula <-
    'cbind(' %.%                # outcomes
    paste(family.outcomes,
          collapse = ', '
    ) %.%  ') ~\n0 +\n' %.%
    paste(treatments,           # treatments (base terms)
          collapse = ' +\n'
    ) %.% ' +\n' %.%
    paste(family.controls,      # controls (base terms)
          collapse = ' +\n'
    )##  %.% ' +\n' %.%
  ## paste(                      # treat-ctrl interactions
  ##   family.controls.interactions,
  ##   collapse = ' +\n'
  ## )
  
  cat(rep('=', 80),
      '\n\nHYPOTHESIS FAMILY: ',
      family,
      '\n\nrunning mlm:\n\n',
      family.formula,
      '\n\n',
      sep = ''
  )
  
  ## run model
  family.mod <- lm(family.formula, d)
  
  ## hack to eliminate NA coefs
  if (any(is.na(coef(family.mod)))){
    if ('mlm' %in% class(family.mod)){
      drop <- rownames(coef(family.mod))[is.na(coef(family.mod))[, 1]]
    } else {
      drop <- names(coef(family.mod))[is.na(coef(family.mod))]
    }
    drop <- convert.interaction.names(drop,
                                      c(family.controls,
                                        family.controls.interactions
                                      )
    )
    layer1.notes[[i]] <-
      layer1.notes[[i]] %.%
      'dropped the following coefs: ' %.%
      paste(drop, sep = ', ') %.%
      '\n\n'
    family.formula <- gsub(
      '\\s+\\+\\s+(' %.% paste(drop, collapse = '|') %.% ')',
      '',
      family.formula
    )
    family.mod <- lm(family.formula, d)
  }
  
  family.vcov <- vcovHC(family.mod)
  if (is.null(dim(coef(family.mod)))){
    coef.names <- names(coef(family.mod))
  } else {
    coef.names <- rownames(coef(family.mod))
  }
  
  ### top layer: test overall significance of all contrasts on all outcomes ###
  ## convert interaction terms to whatever mlm() named it
  treats <- convert.interaction.names(contrasts[, 'treat'], coef.names)
  ctrls <- convert.interaction.names(contrasts[, 'ctrl'], coef.names)
  
  ## test jointly
  lht.attempt <- tryCatch({
    if ('mlm' %in% class(family.mod)){
      contrast.lht <- linearHypothesis(
        family.mod,
        vcov. = family.vcov,
        hypothesis.matrix = sprintf('%s - %s', treats, ctrls),
        rhs = matrix(0, nrow = nrow(contrasts), ncol = length(family.outcomes)),
        test = 'Pillai'
      )
      layer1.pvals[[i]] <- extract.lht(contrast.lht)[, 'Pr(>F)']
    } else {
      contrast.lht <- linearHypothesis(
        family.mod,
        vcov. = family.vcov,
        hypothesis.matrix = sprintf('%s - %s', treats, ctrls),
        rhs = matrix(0, nrow = nrow(contrasts), ncol = length(family.outcomes)),
        test = 'F'
      )
      layer1.pvals[[i]] <- contrast.lht[['Pr(>F)']][2]
    }
  },
  error = function(e){
    warning(sprintf('caught error in %s family:', family), e)
    ## return error as string for inclusion in notes
    'caught error: ' %.%
      e %.%
      '\n\n'
  })
  if (lht.attempt %like% 'caught error'){
    layer1.notes[[i]] <-
      layer1.notes[[i]] %.% lht.attempt
  }

  ### layer 2: test each contrast individually on all outcomes ###
  
  for (j in 1:nrow(contrasts)){
    ## test group equality on all outcomes
    if ('mlm' %in% class(family.mod)){
      contrast.lht <-
        linearHypothesis(
          family.mod,
          vcov. = family.vcov,
          hypothesis.matrix = sprintf('%s - %s', treats[j], ctrls[j]),
          rhs = matrix(0, nrow = 1, ncol = length(family.outcomes)),
          test = 'Pillai'
        )
      layer2.pvals[[i]][j] <- extract.lht(contrast.lht)[, 'Pr(>F)']
    } else {
      contrast.lht <- linearHypothesis(
        family.mod,
        vcov. = family.vcov,
        hypothesis.matrix = sprintf('%s - %s', treats[j], ctrls[j]),
        rhs = matrix(0, nrow = 1, ncol = length(family.outcomes)),
        test = 'F'
      )
      layer2.pvals[[i]][j] <- contrast.lht[['Pr(>F)']][2]
    }
  }
  
  ### layer 3: test each contrast on each outcome individually ###
  
  for (k in 1:length(family.outcomes)){
    
    outcome <- family.outcomes[k]
    
    outcome.formula <-
      outcome %.% ' ~\n0 +\n' %.%
      paste(treatments,         # treatments (base terms)
            collapse = ' +\n'
      ) %.% ' +\n' %.%
      paste(family.controls,      # controls (base terms)
            collapse = ' +\n'
      )##  %.% ' +\n' %.%
    ## paste(                      # treat-ctrl interactions
    ##   family.controls.interactions,
    ##   collapse = ' +\n'
    ## )
    
    cat(rep('-', 40), '\n\nrunning lm:\n\n', outcome.formula, '\n\n', sep = '')
    
    outcome.mod <- lm(outcome.formula, d)
    ## hack to eliminate NA coefs
    if (any(is.na(coef(outcome.mod)))){
      drop <- names(coef(outcome.mod))[is.na(coef(outcome.mod))]
      drop <- convert.interaction.names(drop,
                                        c(family.controls,
                                          family.controls.interactions
                                        )
      )
      layer3.notes[[i]][k] <-
        layer3.notes[[i]][k] %.%
        'dropped the following coefs: ' %.%
        paste(drop, sep = ', ') %.%
        '\n\n'
      outcome.formula <- gsub(
        '\\s+\\+\\s+(' %.% paste(drop, collapse = '|') %.% ')',
        '',
        outcome.formula
      )
      outcome.mod <- lm(outcome.formula, d)
    }
    
    outcome.vcov <- vcovHC(outcome.mod)
    if (any(!is.finite(outcome.vcov))){
      outcome.vcov <- vcov(outcome.mod)
      layer3.notes[[i]][k] <-
        layer3.notes[[i]][k] %.%
        'falling back to non-robust vcov\n\n'
    }
    coef.names <- names(coef(outcome.mod))
    
    for (j in 1:nrow(contrasts)){
      
      ## convert this interaction term to whatever llm() named it
      treat <- convert.interaction.names(contrasts[j, 'treat'], coef.names)
      ctrl <- convert.interaction.names(contrasts[j, 'ctrl'], coef.names)
      ## test group equality on this outcome
  
      
      contrast.lht <- linearHypothesis(
        outcome.mod,
        vcov. = outcome.vcov,
        hypothesis.matrix = sprintf('%s - %s', treat, ctrl),
        test = 'F'
      )
      
      layer3.pvals[[i]][[j]][k] <- contrast.lht[['Pr(>F)']][2]
      layer3.ests[[i]][[j]][k] <- (
        coef(outcome.mod)[treat] - coef(outcome.mod)[ctrl]
      ) ## * attr(d[[outcome]], 'scaled:scale')  # note: uncomment if rescaling
      layer3.ses[[i]][[j]][k] <- sqrt(
        outcome.vcov[treat, treat] +
          outcome.vcov[ctrl, ctrl] -
          2 * outcome.vcov[treat, ctrl]
      )
    }
  }
}

#################################
## multiple testing correction ##
#################################

thresh <- .05

## if layer-1 f-test is infeasible for a family due to collinearity,
##   obtain layer-1 p-values for that family by simes
for (i in which(is.na(layer1.pvals))){
  layer1.pvals[i] <- simes(layer2.pvals[[i]])
}

## multiple testing adjustment for layer 1
layer1.pvals.adj <- p.adjust(layer1.pvals, 'BH')
layer1.nonnull.prop <- mean(layer1.pvals.adj < thresh)

## test layer-2 hypotheses only if layer 1 passes
layer2.pvals.adj <- layer2.pvals  # start by copying unadjusted layer-2 p-values
layer2.nonnull.prop <- rep(NA, length(layer1.pvals.adj))
names(layer2.nonnull.prop) <- names(layer1.pvals.adj)
for (i in 1:length(layer1.pvals)){
  if (layer1.pvals.adj[i] < thresh){  # if layer 1 passes
    ## adjust for multiplicity within layer 2...
    layer2.pvals.adj[[i]] <- p.adjust(layer2.pvals[[i]], 'BH')
    ## ... and inflate to account for selection at layer 1
    layer2.pvals.adj[[i]] <-
      pmin(layer2.pvals.adj[[i]] / layer1.nonnull.prop, 1)
    ## keep track of selection at layer 2 for use in layer 3
    layer2.nonnull.prop[i] <- mean(layer2.pvals.adj[[i]] < thresh)
  } else {  # if layer 1 fails
    layer2.pvals.adj[[i]] <- rep(NA_real_, length(layer2.pvals[[i]]))
    names(layer2.pvals.adj[[i]]) <- names(layer2.pvals[[i]])
  }
}

## test layer-3 hypotheses only if layers 1 & 2 pass
layer3.pvals.adj <- layer3.pvals  # start by copying unadjusted layer-3 p-values
for (i in 1:length(layer1.pvals.adj)){
  for (j in 1:length(layer2.pvals.adj[[i]])){
    ##
    if (layer1.pvals.adj[i] < thresh &&      # if layer 1 passes...
        layer2.pvals.adj[[i]][j] < thresh  # ... and if layer 2 passes
    ){
      ## adjust for multiplicity within layer 3...
      layer3.pvals.adj[[i]][[j]] <- p.adjust(layer3.pvals[[i]][[j]], 'BH')
      ## ... and inflate to account for selection at layer 1
      layer3.pvals.adj[[i]][[j]] <- pmin(
        layer3.pvals.adj[[i]][[j]] / layer1.nonnull.prop / layer2.nonnull.prop[i],
        1
      )
    } else {
      layer3.pvals.adj[[i]][[j]] <- rep(NA_real_, length(layer3.pvals[[i]][[j]]))
      names(layer3.pvals.adj[[i]][[j]]) <- names(layer3.pvals[[i]][[j]])
    }
  }
}

pvals.adj <- data.table(layer1 = character(0),
                        layer2 = character(0),
                        layer3 = character(0),
                        p.adj = numeric(0),
                        est = numeric(0),
                        se = numeric(0)
)
for (i in 1:length(layer1.pvals.adj)){
  pvals.adj <- rbind(pvals.adj,
                     data.table(layer1 = names(layer1.pvals.adj)[i],
                                layer2 = 'overall',
                                layer3 = 'overall',
                                p.adj = layer1.pvals.adj[i],
                                est = NA_real_,
                                se = NA_real_
                     )
  )
  for (j in 1:length(layer2.pvals.adj[[i]])){
    pvals.adj <- rbind(pvals.adj,
                       data.table(layer1 = names(layer1.pvals.adj)[i],
                                  layer2 = names(layer2.pvals.adj[[i]])[j],
                                  layer3 = 'overall',
                                  p.adj = layer2.pvals.adj[[i]][j],
                                  est = NA_real_,
                                  se = NA_real_
                       )
    )
    for (k in 1:length(layer3.pvals.adj[[i]][[j]])){
      pvals.adj <- rbind(pvals.adj,
                         data.table(layer1 = names(layer1.pvals.adj)[i],
                                    layer2 = names(layer2.pvals.adj[[i]])[j],
                                    layer3 = names(layer3.pvals.adj[[i]][[j]])[k],
                                    p.adj = layer3.pvals.adj[[i]][[j]][k],
                                    est = layer3.ests[[i]][[j]][k],
                                    se = layer3.ses[[i]][[j]][k]
                         )
      )
    }
  }
}

## write out
fwrite(pvals.adj, '../results/padj_basecontrol_may2024.csv')


## prettify for reading
pvals.adj.pretty <- pvals.adj
colnames(pvals.adj.pretty) <- gsub('layer1',
                                   'layer1_hypothesisfamily',
                                   colnames(pvals.adj.pretty)
)
colnames(pvals.adj.pretty) <- gsub('layer2',
                                   'layer2_treatmentcontrast',
                                   colnames(pvals.adj.pretty)
)
colnames(pvals.adj.pretty) <- gsub('layer3',
                                   'layer3_specificoutcome',
                                   colnames(pvals.adj.pretty)
)

pvals.adj.pretty[, layer2_treatmentcontrast := gsub(
  'attitude\\.(pro|anti|neutral)(:assg\\.(inc|cons))?:recsys.(ca|cp|ip|ia)',
  '\\1 \\3 \\4',
  layer2_treatmentcontrast
)]
pvals.adj.pretty[, layer2_treatmentcontrast := gsub(
  '.vs.',
  ' - ',
  layer2_treatmentcontrast,
  fixed = TRUE
)]
pvals.adj.pretty[, layer2_treatmentcontrast := gsub(
  ' +',
  ' ',
  layer2_treatmentcontrast
)]
fwrite(pvals.adj.pretty,
       '../results/padj_basecontrol_pretty_ytrecs_may2024.csv'
)

# pvals.adj.pretty[p.adj < .05 & layer3_specificoutcome != 'overall',]

################################
######### OMNIBUS TEST #########
################################

# Step 1: Create a binary variable indicating increasing condition
d$is_increasing <- ifelse(d$treatment_arm == "pi" | d$treatment_arm == "ai", 1, 0)

# Step 2: Reverse values for individuals in the Pro condition
d$mw_index_pre[d$treatment_arm %like% "pi|pc"] <- 1 - d$mw_index_pre[d$treatment_arm %like% "pi|pc"]
d$mw_index[d$treatment_arm %like% "pi|pc"] <- 1 - d$mw_index[d$treatment_arm %like% "pi|pc"]

# Step 3: Perform the linear regression (omnibus test)
model <- lm(I(mw_index - mw_index_pre) ~ is_increasing, data = d)

# View the summary of the model
summary(model)
rm(list = ls())
