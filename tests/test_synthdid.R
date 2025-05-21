library(synthdid)

# Estimate the effect of California Proposition 99 on cigarette consumption
data('CPS')
setup = panel.matrices(CPS)
tau.hat = synthdid_estimate(setup$Y, setup$N0, setup$T0)
se = sqrt(vcov(tau.hat, method='placebo'))
sprintf('point estimate: %1.2f', tau.hat)
sprintf('95%% CI (%1.2f, %1.2f)', tau.hat - 1.96 * se, tau.hat + 1.96 * se)
plot(tau.hat)
