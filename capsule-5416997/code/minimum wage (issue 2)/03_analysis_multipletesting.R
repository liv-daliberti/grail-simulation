cat(rep('=', 80),
    '\n\n',
    'OUTPUT FROM: minimum wage (issue 2)/03_analysis_multipletesting.R',
    '\n\n',
    sep = ''
    )

library(data.table)
library(car)
library(sandwich)
library(lmtest)
library(ggplot2)
library(assertthat)
library(foreach)
library(doParallel)
registerDoParallel(cores = detectCores() - 1)



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

d <- fread('../results/intermediate data/minimum wage (issue 2)/qualtrics_w12_clean.csv')

## drop pure control
d <- d[treatment_arm != 'control',]

## drop NA video counts
d <- d[!is.na(pro) & !is.na(anti),]



##############
## controls ##
##############

platform.controls <- c('age_cat',
                       'male',
                       'pol_interest',
                       'freq_youtube'
                       )

mwpolicy.controls <- 'mw_index_w1'

media.controls <- c('trust_majornews_w1',
                    'trust_youtube_w1',
                    'fabricate_majornews_w1',
                    'fabricate_youtube_w1'
                    )

affpol.controls <- c('affpol_ft',
                     'affpol_smart',
                     'affpol_comfort'
                     )

controls.raw <- unique(c(platform.controls,
                         mwpolicy.controls,
                         media.controls,
                         affpol.controls
                         )
                       )

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



##############
## outcomes ##
##############

### hypothesis family 1: platform interactions ###

## platform interaction time: compute windorized usage time
warning('diverges from pap, 95% windsorized due to extreme outliers')
d[, platform_duration := duration]
d[platform_duration <= quantile(d$duration, .025),
  platform_duration := quantile(d$duration, .025)
  ]
d[platform_duration >= quantile(d$duration, .975),
  platform_duration := quantile(d$duration, .975)
  ]
## all platform interaction outcomes
platform.outcomes <- c('pro_fraction_chosen',
                       'positive_interactions',  # positive - negative (dislike)
                       'platform_duration'
                       )



### hypothesis family 2: MW policy attitudes ###

## only one preregistered outcome in this family
mwpolicy.outcomes <- 'mw_index_w2'
## added 4 jun 2024 at request of reviewers
mwpolicy.outcomes.understanding <- c('mw_restrict_w2',
                                     'mw_help_w2'
                                     )



### hypothesis family 3: media trust ###
media.outcomes <- c('trust_majornews_w2',
                    'trust_youtube_w2',
                    'fabricate_majornews_w2',
                    'fabricate_youtube_w2'
                    )



### hypothesis family 4: affective polarization ###
affpol.outcomes <- c('affpol_ft_w2',
                     'affpol_smart_w2',
                     'affpol_comfort_w2'
                     )

outcomes <- unique(c(
  platform.outcomes,
  mwpolicy.outcomes,
  media.outcomes,
  affpol.outcomes
)
)



################
## treatments ##
################

## create attitude dummies
## (pro/anti stance on issue has opposite lib/con meaning from study 1)
d[, attitude := c('pro', 'neutral', 'anti')[thirds]]
d[, attitude.anti := as.numeric(attitude == 'anti')]
d[, attitude.neutral := as.numeric(attitude == 'neutral')]
d[, attitude.pro := as.numeric(attitude == 'pro')]

## create seed dummies
d[, seed.anti := as.numeric(treatment_seed == 'anti')]
d[, seed.pro := as.numeric(treatment_seed == 'pro')]

## create recsys dummies
d[, recsys.22 := as.numeric(treatment_arm %like% '22')]
d[, recsys.31 := as.numeric(treatment_arm %like% '31')]

## manually define coefficients to estimate
treatments <- c('attitude.anti:recsys.22',
                'attitude.anti:recsys.31',
                'attitude.neutral:seed.anti:recsys.22',
                'attitude.neutral:seed.pro:recsys.22',
                'attitude.neutral:seed.anti:recsys.31',
                'attitude.neutral:seed.pro:recsys.31',
                'attitude.pro:recsys.22',
                'attitude.pro:recsys.31'
                )

contrasts <- rbind(
  i = c(treat = 'attitude.pro:recsys.31',
        ctrl = 'attitude.pro:recsys.22'
        ),
  ii = c(treat = 'attitude.anti:recsys.31',
         ctrl = 'attitude.anti:recsys.22'
         ),
  iii = c(treat = 'attitude.neutral:seed.pro:recsys.31',
          ctrl = 'attitude.neutral:seed.pro:recsys.22'
          ),
  iv = c(treat = 'attitude.neutral:seed.anti:recsys.31',
         ctrl = 'attitude.neutral:seed.anti:recsys.22'
         ),
  # in (v-vi), pro/anti order is reversed from study 1 to ensure that
  # - 1st condition (treatment) is always the conservative video
  # - 2nd condition (control) is always the liberal video
  v = c(treat = 'attitude.neutral:seed.anti:recsys.31',
        ctrl = 'attitude.neutral:seed.pro:recsys.31'
  ),
  vi = c(treat = 'attitude.neutral:seed.anti:recsys.22',
         ctrl = 'attitude.neutral:seed.pro:recsys.22'
  )
)

## check that contrasts are valid
assert_that(all(unlist(contrasts) %in% treatments))

## check that specifications are equivalent
coefs.v1 <- coef(lm(mw_index_w2 ~ 0 + attitude:treatment_arm, d))
coefs.v2 <- coef(
  lm(mw_index_w2 ~
       0 +
       attitude.anti:recsys.22 +
       attitude.anti:recsys.31 +
       attitude.neutral:seed.anti:recsys.22 +
       attitude.neutral:seed.pro:recsys.22 +
       attitude.neutral:seed.anti:recsys.31 +
       attitude.neutral:seed.pro:recsys.31 +
       attitude.pro:recsys.22 +
       attitude.pro:recsys.31,
     d
     )
)
assert_that(all.equal(unname(sort(coefs.v1)), unname(sort(coefs.v2))))

##########################
## hierarchical testing ##
##########################

## initialize top layer p-values:
##   does treatment have any effect on any outcome in family
families <- c(
  'platform',
  'mwpolicy',
  'media',
  'affpol'
)
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
layer2.pvals <- list(
  platform = contrast.pvals,
  mwpolicy = contrast.pvals,
  media = contrast.pvals,
  affpol = contrast.pvals
)
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
          )

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
          )

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

      ## ## confirm
      ## linearHypothesis(
      ##   outcome.mod,
      ##   vcov. = outcome.vcov,
      ##   hypothesis.matrix = sprintf('%s - %s', treat, ctrl),
      ##   test = 'F'
      ## )
      ## (coef(outcome.mod)[treat] - coef(outcome.mod)[ctrl])^2 /
      ##   (
      ##     outcome.vcov[treat, treat] +
      ##       outcome.vcov[ctrl, ctrl] -
      ##       2 * outcome.vcov[treat, ctrl]
      ##   )
      ## linearHypothesis(
      ##   outcome.mod,
      ##   vcov. = outcome.vcov,
      ##   hypothesis.matrix = sprintf('%s - %s', treat, ctrl),
      ##   test = 'Chisq'
      ## )
      ## 2 - 2 * pnorm(abs(
      ## (coef(outcome.mod)[treat] - coef(outcome.mod)[ctrl]) /
      ##   sqrt(
      ##     outcome.vcov[treat, treat] +
      ##       outcome.vcov[ctrl, ctrl] -
      ##       2 * outcome.vcov[treat, ctrl]
      ##   )
      ## ))

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
fwrite(pvals.adj, '../results/intermediate data/minimum wage (issue 2)/padj_basecontrol.csv')

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
  'attitude\\.(pro|anti|neutral)(:seed\\.(pro|anti))?:recsys.(31|22)',
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
       '../results/intermediate data/minimum wage (issue 2)/padj_basecontrol_pretty.csv'
       )

print('preregistered results:')
pvals.adj.pretty[p.adj < .05 & layer3_specificoutcome != 'overall',]



##############################################
## added 4 jun 2024 at request of reviewers ##
##############################################

## analyze components of main policy outcome index that relate to
## post-experiment w2 "understanding" of an issue, using w1 version
## of that same outcome as the only control (analogous to outcome index
## regression, which uses w2 index as outcome and w1 index as control)

## initialize results table
understanding.results <- data.table(layer2_treatmentcontrast = character(0),
                                    layer3_specificoutcome = character(0),
                                    est = numeric(0),
                                    se = numeric(0),
                                    p = numeric(0)
                                    )

## loop over outcomes
for (k in 1:length(mwpolicy.outcomes.understanding)){

  outcome <- mwpolicy.outcomes.understanding[k]

  outcome.formula <-
    outcome %.% ' ~\n0 +\n' %.%
    paste(treatments,         # treatments (base terms)
          collapse = ' +\n'
          ) %.% ' +\n' %.%
    paste(gsub('w2', 'w1', outcome),      # controls (w1 outcome)
          collapse = ' +\n'
          )

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

  ## loop over treatment contrasts
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

    ## prettify name of contrast for readability

    contrast <- treat %.% ' - ' %.% ctrl
    contrast <- gsub('attitude\\.(pro|anti|neutral)', '\\1', contrast)
    contrast <- gsub('seed\\.(pro|anti)', '\\1', contrast)
    contrast <- gsub('recsys.(31|22)', '\\1', contrast)
    contrast <- gsub(':', ' ', contrast)
    contrast <- gsub(' +', ' ', contrast)

    p <- contrast.lht[['Pr(>F)']][2]
    est <- (
      coef(outcome.mod)[treat] - coef(outcome.mod)[ctrl]
    ) ## * attr(d[[outcome]], 'scaled:scale')  # note: uncomment if rescaling
    se <- sqrt(
      outcome.vcov[treat, treat] +
        outcome.vcov[ctrl, ctrl] -
        2 * outcome.vcov[treat, ctrl]
    )

    understanding.results <- rbind(
      understanding.results,
      data.table(
        layer2_treatmentcontrast = contrast,
        layer3_specificoutcome = outcome,
        p,
        est,
        se
      )
    )

  }

}

## conduct multiple testing adjustment within newly exploratory results
understanding.results[, p.adj := p.adjust(p, 'BH')]
print('exploratory results on understanding-related questions:')
understanding.results[p.adj < .05,]
fwrite(understanding.results,
       '../results/intermediate data/minimum wage (issue 2)/understanding_basecontrol_pretty.csv'
       )



#############################################################
## preregistered exploratory heterogeneous effect analysis ##
#############################################################

# outcome is mw_index_w2
# construct moderators by cutting demographics & pre-treatment vars at midpoint

d[,
  pol_interest_hi := as.numeric(
    pol_interest > median(pol_interest, na.rm = TRUE)
  )]
d[,
  age_hi := as.numeric(
    age > median(age, na.rm = TRUE)
  )]
d[,
  freq_youtube_hi := as.numeric(
    freq_youtube > median(freq_youtube, na.rm = TRUE)
  )]

moderator_variables <- c('pol_interest_hi',
                         'age_hi',
                         'male',
                         'freq_youtube_hi'
                         )
## added 4 jun 2024 at request of reviewer
moderator_variables_revision <- 'college'

interaction_results <- data.table()
for (moderator_variable in c(moderator_variables, moderator_variables_revision)){

  d[, moderator := get(moderator_variable)]

  mod.attitude.anti <- lm(
    mw_index_w2 ~
      recsys.31 * moderator +
      mw_index_w1,   # only control is pre-treatment outcome, as in primary analysis
    data = d[attitude.anti == 1]
  )
  vcov.attitude.anti <- vcovHC(mod.attitude.anti)
  test.attitude.anti <- coeftest(mod.attitude.anti, vcov.attitude.anti)
  interaction_results <- rbind(
    interaction_results,
    data.table(subset = 'attitude.anti',
               interaction = 'recsys.31:' %.% moderator_variable,
               test.attitude.anti['recsys.31:moderator', , drop = FALSE]
               ),
    fill = TRUE
  )

  mod.attitude.pro <- lm(
    mw_index_w2 ~
      recsys.31 * moderator +
      mw_index_w1,   # only control is pre-treatment outcome, as in primary analysis
    data = d[attitude.pro == 1]
  )
  vcov.attitude.pro <- vcovHC(mod.attitude.pro)
  test.attitude.pro <- coeftest(mod.attitude.pro, vcov.attitude.pro)
  interaction_results <- rbind(
    interaction_results,
    data.table(subset = 'attitude.pro',
               interaction = 'recsys.31:' %.% moderator_variable,
               test.attitude.pro['recsys.31:moderator', , drop = FALSE]
               ),
    fill = TRUE
  )

  mod.attitude.neutral.seed.anti <- lm(
    mw_index_w2 ~
      recsys.31 * moderator +
      mw_index_w1,   # only control is pre-treatment outcome, as in primary analysis
    data = d[attitude.neutral == 1 & seed.anti == 1]
  )
  vcov.attitude.neutral.seed.anti <- vcovHC(mod.attitude.neutral.seed.anti)
  test.attitude.neutral.seed.anti <- coeftest(mod.attitude.neutral.seed.anti,
                                             vcov.attitude.neutral.seed.anti
                                             )
  interaction_results <- rbind(
    interaction_results,
    data.table(subset = 'attitude.neutral.seed.anti',
               interaction = 'recsys.31:' %.% moderator_variable,
               test.attitude.neutral.seed.anti[
                 'recsys.31:moderator', , drop = FALSE
               ]
               ),
    fill = TRUE
  )

  mod.attitude.neutral.seed.pro <- lm(
    mw_index_w2 ~
      recsys.31 * moderator +
      mw_index_w1,   # only control is pre-treatment outcome, as in primary analysis
    data = d[attitude.neutral == 1 & seed.pro == 1]
  )
  vcov.attitude.neutral.seed.pro <- vcovHC(mod.attitude.neutral.seed.pro)
  test.attitude.neutral.seed.pro <- coeftest(mod.attitude.neutral.seed.pro,
                                             vcov.attitude.neutral.seed.pro                                             )
  interaction_results <- rbind(
    interaction_results,
    data.table(subset = 'attitude.neutral.seed.pro',
               interaction = 'recsys.31:' %.% moderator_variable,
               test.attitude.neutral.seed.pro[
                 'recsys.31:moderator', , drop = FALSE
               ]
               ),
    fill = TRUE
  )

}

# very little significant heterogeneity even before multiple testing correction
# out of 16 tests, 2 have p values of .043 and .032
print('heterogeneity results before multiple correction:')
interaction_results[`Pr(>|t|)` < .05,]
# none survives a BH correction
interaction_results[, p.adj := p.adjust(`Pr(>|t|)`, 'BH')]
print('heterogeneity p-values after multiple correction:')
interaction_results[, p.adj]

## added 4 jun 2024 at request of reviewers
colnames(interaction_results) <- c(
  subset = 'subset',
  interaction = 'interaction',
  Estimate = 'est',
  `Std. Error` = 'se',
  `t value` = 't',
  `Pr(>|t|)` = 'p',
  p.adj = 'p.adj'
)[colnames(interaction_results)]
fwrite(interaction_results,
        '../results/intermediate data/minimum wage (issue 2)/heterogeneity_basecontrol.csv'
        )



###############################################
## added 30 sep 2024 at request of reviewers ##
###############################################

## what are minimum detectable effects, given multiple testing correction?

n_sims <- 1000
params_sims <- expand.grid(seed = 19104 + 0:(n_sims - 1),
                           effect = seq(from = .01, to = .05, by = .001)
                           )

## step 1: identify largest p-value s.t. we would have rejected layer-1 null
## (that at least one treatment contrast has effect on policy index)
## to do this, we hold fixed p-values for all other layer-1 hypothesis families
layer1.pvals.mde <- layer1.pvals
layer1.pvals.mde['mwpolicy'] <- 0
while (p.adjust(layer1.pvals.mde, 'BH')['mwpolicy'] <= .05){
  layer1.pvals.mde['mwpolicy'] <- layer1.pvals.mde['mwpolicy'] + .001
}
pval.cutoff <- layer1.pvals.mde['mwpolicy']
print('to achieve significance of policy attitude family at layer 1 (pooled test of any effect on policy index from any contrast) when correcting for multiple layer-1 hypothesis families, this is the minimum cutoff value after conducting simes correction of layer 2 pvals:')
pval.cutoff

## if layer-1 null was rejected for the policy outcome, then we would use this
## correction factor when interpreting layer-2 p-values (for specific contrasts)
layer1.nonnull.prop.if.gt.cutoff <- mean(c(
  p.adjust(layer1.pvals.mde, 'BH')[c('platform', 'media', 'affpol')] < .05,
  TRUE
))

## the sims below will only examine 3/1 vs 2/2 treatment contrasts, so we will
## hold fixed the layer-2 p-values that relate to seed contrasts
pvals.for.seed.contrasts.on.policyindex <- layer2.pvals$mwpolicy[
  c('attitude.neutral:seed.pro:recsys.31.vs.attitude.neutral:seed.anti:recsys.31',
    'attitude.neutral:seed.pro:recsys.22.vs.attitude.neutral:seed.anti:recsys.22'
    )
]



## step 2: prepare simulations based on real data ------------------------------

mod.attitude.anti <- lm(
  mw_index_w2 ~ recsys.31 + mw_index_w1,
  data = d[attitude.anti == 1]
)
X.attitude.anti <- model.matrix(mod.attitude.anti)
residual.sd.attitude.anti <- sd(resid(mod.attitude.anti))
## confirm that this recovers fitted values
##   model.matrix(mod.attitude.anti) %*% coef(mod.attitude.anti)
assert_that(all(
  predict(mod.attitude.anti) ==
    X.attitude.anti %*% coef(mod.attitude.anti)
))
## we will create simulated outcomes, given hypothesized treatment effect
## == intercept +                                              <-- part A
##    real coef * real pretreatment attitude +                 <-- part A
##    hypothesized treatment effect * real treatment status +  <-- part B
##    rnorm(mean = 0, sd = real residual outcome sd)           <-- part C
## A: generate fitted values under hypothesized effect size
coef.attitude.anti.baseline <- coef(mod.attitude.anti)
coef.attitude.anti.baseline['recsys.31'] <- 0
Y.attitude.anti.baseline <-
  as.numeric(X.attitude.anti %*% coef.attitude.anti.baseline)
## C: will be added below with hypothesized effect * treatment
## B: will be drawn below with rnorm(mean=0, sd=residual_sd)

## repeat above for respondents with pro attitude
mod.attitude.pro <- lm(
  mw_index_w2 ~ recsys.31 + mw_index_w1,
  data = d[attitude.pro == 1]
)
X.attitude.pro <- model.matrix(mod.attitude.pro)
residual.sd.attitude.pro <- sd(resid(mod.attitude.pro))
coef.attitude.pro.baseline <- coef(mod.attitude.pro)
coef.attitude.pro.baseline['recsys.31'] <- 0
Y.attitude.pro.baseline <-
  as.numeric(X.attitude.pro %*% coef.attitude.pro.baseline)

## repeat above for respondents with neutral attitude assigned to pro seed
mod.attitude.neutral.seed.pro <- lm(
  mw_index_w2 ~ recsys.31 + mw_index_w1,
  data = d[attitude.neutral == 1 & seed.pro == 1]
)
X.attitude.neutral.seed.pro <- model.matrix(mod.attitude.neutral.seed.pro)
residual.sd.attitude.neutral.seed.pro <- sd(resid(mod.attitude.neutral.seed.pro))
coef.attitude.neutral.seed.pro.baseline <- coef(mod.attitude.neutral.seed.pro)
coef.attitude.neutral.seed.pro.baseline['recsys.31'] <- 0
Y.attitude.neutral.seed.pro.baseline <-
  as.numeric(X.attitude.neutral.seed.pro %*% coef.attitude.neutral.seed.pro.baseline)

## repeat above for respondents with neutral attitude assigned to anti seed
mod.attitude.neutral.seed.anti <- lm(
  mw_index_w2 ~ recsys.31 + mw_index_w1,
  data = d[attitude.neutral == 1 & seed.anti == 1]
)
X.attitude.neutral.seed.anti <- model.matrix(mod.attitude.neutral.seed.anti)
residual.sd.attitude.neutral.seed.anti <- sd(resid(mod.attitude.neutral.seed.anti))
coef.attitude.neutral.seed.anti.baseline <- coef(mod.attitude.neutral.seed.anti)
coef.attitude.neutral.seed.anti.baseline['recsys.31'] <- 0
Y.attitude.neutral.seed.anti.baseline <-
  as.numeric(X.attitude.neutral.seed.anti %*% coef.attitude.neutral.seed.anti.baseline)



## step 3: conduct sims --------------------------------------------------------

sims.attitude.anti <- foreach(seed = params_sims$seed,
                              effect = params_sims$effect,
                              .combine = rbind
                              ) %dopar%
  {
    set.seed(seed)
    Y <-
      Y.attitude.anti.baseline +
      effect * X.attitude.anti[, 'recsys.31'] +
      rnorm(
        n = nrow(X.attitude.anti),
        mean = 0,
        sd = residual.sd.attitude.anti
      )
    mod <- lm(Y ~ 0 + X.attitude.anti)
    smry <- coeftest(mod, vcovHC(mod))
    cbind(
      seed,
      effect,
      data.table(smry['X.attitude.antirecsys.31', , drop = FALSE])
    )
  }

sims.attitude.pro <- foreach(seed = params_sims$seed,
                              effect = params_sims$effect,
                              .combine = rbind
                              ) %dopar%
  {
    set.seed(seed)
    Y <-
      Y.attitude.pro.baseline +
      effect * X.attitude.pro[, 'recsys.31'] +
      rnorm(
        n = nrow(X.attitude.pro),
        mean = 0,
        sd = residual.sd.attitude.pro
      )
    mod <- lm(Y ~ 0 + X.attitude.pro)
    smry <- coeftest(mod, vcovHC(mod))
    cbind(
      seed,
      effect,
      data.table(smry['X.attitude.prorecsys.31', , drop = FALSE])
    )
  }

sims.attitude.neutral.seed.anti <- foreach(seed = params_sims$seed,
                                           effect = params_sims$effect,
                                           .combine = rbind
                                           ) %dopar%
  {
    set.seed(seed)
    Y <-
      Y.attitude.neutral.seed.anti.baseline +
      effect * X.attitude.neutral.seed.anti[, 'recsys.31'] +
      rnorm(
        n = nrow(X.attitude.neutral.seed.anti),
        mean = 0,
        sd = residual.sd.attitude.neutral.seed.anti
      )
    mod <- lm(Y ~ 0 + X.attitude.neutral.seed.anti)
    smry <- coeftest(mod, vcovHC(mod))
    cbind(
      seed,
      effect,
      data.table(smry['X.attitude.neutral.seed.antirecsys.31', , drop = FALSE])
    )
  }

sims.attitude.neutral.seed.pro <- foreach(seed = params_sims$seed,
                                           effect = params_sims$effect,
                                           .combine = rbind
                                           ) %dopar%
  {
    set.seed(seed)
    Y <-
      Y.attitude.neutral.seed.pro.baseline +
      effect * X.attitude.neutral.seed.pro[, 'recsys.31'] +
      rnorm(
        n = nrow(X.attitude.neutral.seed.pro),
        mean = 0,
        sd = residual.sd.attitude.neutral.seed.pro
      )
    mod <- lm(Y ~ 0 + X.attitude.neutral.seed.pro)
    smry <- coeftest(mod, vcovHC(mod))
    cbind(
      seed,
      effect,
      data.table(smry['X.attitude.neutral.seed.prorecsys.31', , drop = FALSE])
    )
  }



## step 4: analyze power results -----------------------------------------------

## without multiple-testing corrections

print('mde for respondents with anti attitude (conventional analysis w/o correction):')
sims.attitude.anti[,
                   .(p.reject = mean(`Pr(>|t|)` < .05)),
                   by = effect
                   ][p.reject >= .8, min(effect)]

print('mde for respondents with pro attitude (conventional analysis w/o correction):')
sims.attitude.pro[,
                  .(p.reject = mean(`Pr(>|t|)` < .05)),
                  by = effect
                  ][p.reject >= .8, min(effect)]

print('mde for respondents with neutral attitude assigned to pro seed (conventional analysis w/o correction):')
sims.attitude.neutral.seed.anti[,
                                .(p.reject = mean(`Pr(>|t|)` < .05)),
                                by = effect
                                ][p.reject >= .8, min(effect)]

## respondents with neutral attitude assigned to anti seed
sims.attitude.neutral.seed.pro[,
                               .(p.reject = mean(`Pr(>|t|)` < .05)),
                               by = effect
                               ][p.reject >= .8, min(effect)]



## with multiple testing correction

sims <- rbind(
  sims.attitude.anti,
  sims.attitude.pro,
  sims.attitude.neutral.seed.anti,
  sims.attitude.neutral.seed.pro
)

sims.layer1 <- sims[
,
  .(pval.pooled = ifelse(
    ## if these results would lead us to reject layer-1 pooled null of no effect
    ## on policy attitudes from any treatment contrast
    simes(c(
      `Pr(>|t|)`,
      pvals.for.seed.contrasts.on.policyindex
    )) <= pval.cutoff,
    ## disaggregate layer-2 results report with procedure from above
    ## (BH correction, then inflate by 1/prop of layer-1 sig results)
    ## then subset to only those p-values relating to 3/1 vs 2/2 contrast
    ## to see if any are <.05 after full correction procedure
    yes = min(
      p.adjust(c(`Pr(>|t|)`, pvals.for.seed.contrasts.on.policyindex),
               'BH'
               )[1:4] / layer1.nonnull.prop.if.gt.cutoff
    ),
    no = Inf
  )
  ),
  by = .(seed, effect)
]
print('with multiple testing correction:')
sims.layer1[, .(p.reject = mean(pval.pooled <= pval.cutoff)), by = effect]
print('mde:')
sims.layer1[,
            .(p.reject = mean(pval.pooled <= pval.cutoff)),
            by = effect
            ][p.reject >= .8, min(effect)]

