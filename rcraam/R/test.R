
small.test <- function(nature = "l1"){
    mdp.frame <- example_mdp("")

    nature.par <- data.frame( idstate = mdp.frame$idstatefrom,
                idaction = mdp.frame$idaction,
                value = rep(1.0, length(mdp.frame$idaction)))

    rsolve_mdp_sa(mdp.frame, 0.99, nature, nature.par, "mpi")
}
