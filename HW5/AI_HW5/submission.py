"""
Licensing Information: Please do not distribute or publish solutions to this
project. You are free to use and extend Driverless Car for educational
purposes. The Driverless Car project was developed at Stanford, primarily by
Chris Piech (piech@cs.stanford.edu). It was inspired by the Pacman projects.
"""
import collections
import math
import random
import util
from engine.const import Const
from util import Belief


# Class: ExactInference
# ---------------------
# Maintain and update a belief distribution over the probability of a car
# being in a tile using exact updates (correct, but slow times).
class ExactInference(object):
    # Function: Init
    # --------------
    # Constructor that initializes an ExactInference object which has
    # numRows x numCols number of tiles.
    def __init__(self, numRows: int, numCols: int):
        self.skipElapse = False  ### ONLY USED BY GRADER.PY
        # util.Belief is a class (constructor) that represents the belief for a single
        # inference state of a single car (see util.py).
        self.belief = util.Belief(numRows, numCols)
        self.transProb = util.loadTransProb()

    ##################################################################################
    # Part 1:
    # Function: Observe (update the probabilities based on an observation)
    # -----------------
    # Takes |self.belief| -- an object of class Belief, defined in util.py --
    # and updates it in place based on the distance observation $d_t$ and
    # your position $a_t$.
    #
    # - agentX: x location of your car (not the one you are tracking)
    # - agentY: y location of your car (not the one you are tracking)
    # - observedDist: true distance plus a mean-zero Gaussian with standard
    #                 deviation Const.SONAR_STD
    #
    # Notes:
    # - Convert row and col indices into locations using util.rowToY and util.colToX.
    # - util.pdf: computes the probability density function for a Gaussian
    # - Although the gaussian pdf is symmetric with respect to the mean and value,
    #   you should pass arguments to util.pdf in the correct order
    # - Don't forget to normalize self.belief after you update its probabilities!
    ##################################################################################

    def observe(self, agentX: int, agentY: int, observedDist: float) -> None:
        # BEGIN_YOUR_CODE
        """
        1. Compute the distance between your position and the observation.
        2. Use util.pdf() to compute the probability density for a Gaussian where its mean is the computed distance.
        3. Set the probability of each tile to be its pdf times the probability of that tile.
        4. Normalize self.belief to make the all probabilities sum to 1.
        """
        for row in range(self.belief.getNumRows()):
            for col in range(self.belief.getNumCols()):
                dist = ((util.colToX(col) - agentX) ** 2 + (util.rowToY(row) - agentY) ** 2) ** 0.5
                self.belief.setProb(
                    row,
                    col,
                    util.pdf(dist, Const.SONAR_STD, observedDist) * self.belief.getProb(row, col),
                )
        self.belief.normalize()
        # END_YOUR_CODE

    ##################################################################################
    # Part 2:
    # Function: Elapse Time (propose a new belief distribution based on a learned transition model)
    # ---------------------
    # Takes |self.belief| and updates it based on the passing of one time step.
    # Notes:
    # - Use the transition probabilities in self.transProb, which is a dictionary
    #   containing all the ((oldTile, newTile), transProb) key-val pairs that you
    #   must consider.
    # - If there are ((oldTile, newTile), transProb) pairs not in self.transProb,
    #   they are assumed to have zero probability, and you can safely ignore them.
    # - Use the addProb (or setProb) and getProb methods of the Belief class to modify
    #   and access the probabilities associated with a belief.  (See util.py.)
    # - Be careful that you are using only the CURRENT self.belief distribution to compute
    #   updated beliefs.  Don't incrementally update self.belief and use the updated value
    #   for one grid square to compute the update for another square.
    # - Don't forget to normalize self.belief after all probabilities have been updated!
    #   (so that the sum of probabilities is exactly 1 as otherwise adding/multiplying
    #    small floating point numbers can lead to sum being close to but not equal to 1)
    ##################################################################################
    def elapseTime(self) -> None:
        if self.skipElapse:  ### ONLY FOR THE GRADER TO USE IN Part 1
            return
        # BEGIN_YOUR_CODE
        """
        1. Create a new belief which has the same row and column as self.belief.
        2. For each items in self.transProb, compute the transProb times the probability of oldTile.
        3. Increase the probability of newTile by the computed value.
        4. Normalize the new belief and update self.belief to the new belief.
        """
        new_belief = util.Belief(self.belief.getNumRows(), self.belief.getNumCols(), 0)
        for (old, new), prob in self.transProb.items():
            delta = prob * self.belief.getProb(old[0], old[1])
            new_belief.addProb(new[0], new[1], delta)
        new_belief.normalize()
        self.belief = new_belief
        # END_YOUR_CODE

    # Function: Get Belief
    # ---------------------
    # Returns your belief of the probability that the car is in each tile. Your
    # belief probabilities should sum to 1.
    def getBelief(self) -> Belief:
        return self.belief


# Class: Particle Filter
# ----------------------
# Maintain and update a belief distribution over the probability of a car
# being in a tile using a set of particles.
class ParticleFilter(object):
    NUM_PARTICLES = 200

    # Function: Init
    # --------------
    # Constructor that initializes an ParticleFilter object which has
    # (numRows x numCols) number of tiles.
    def __init__(self, numRows: int, numCols: int):
        self.belief = util.Belief(numRows, numCols)

        # Load the transition probabilities and store them in an integer-valued defaultdict.
        # Use self.transProbDict[oldTile][newTile] to get the probability of transitioning
        # from oldTile to newTile.
        self.transProb = util.loadTransProb()
        self.transProbDict = dict()
        for oldTile, newTile in self.transProb:
            if oldTile not in self.transProbDict:
                self.transProbDict[oldTile] = collections.defaultdict(int)
            self.transProbDict[oldTile][newTile] = self.transProb[(oldTile, newTile)]

        # Initialize the particles randomly.
        self.particles = collections.defaultdict(int)
        potentialParticles = list(self.transProbDict.keys())
        for _ in range(self.NUM_PARTICLES):
            particleIndex = int(random.random() * len(potentialParticles))
            self.particles[potentialParticles[particleIndex]] += 1

        self.updateBelief()

    # Function: Update Belief
    # ---------------------
    # Updates |self.belief| with the probability that the car is in each tile
    # based on |self.particles| (which is a defaultdict from grid locations to
    # number of particles at that location) and ensures that the probabilites sum to 1
    def updateBelief(self) -> None:
        newBelief = util.Belief(self.belief.getNumRows(), self.belief.getNumCols(), 0)
        for tile in self.particles:
            newBelief.setProb(tile[0], tile[1], self.particles[tile])
        newBelief.normalize()
        self.belief = newBelief

    ##################################################################################
    # Part 3-1:
    # Function: Observe:
    # -----------------
    # Takes |self.particles| and updates them based on the distance observation
    # $d_t$ and your position $a_t$.
    #
    # This algorithm takes two steps:
    # 1. Re-weight the particles based on the observation.
    #    Concept: We had an old distribution of particles, and now we want to
    #             update this particle distribution with the emission probability
    #             associated with the observed distance.
    #             Think of the particle distribution as the unnormalized posterior
    #             probability where many tiles would have 0 probability.
    #             Tiles with 0 probabilities (i.e. those with no particles)
    #             do not need to be updated.
    #             This makes particle filtering runtime to be O(|particles|).
    #             By comparison, the exact inference method (used in Part 1 + 2)
    #             assigns non-zero (though often very small) probabilities to most tiles,
    #             so the entire grid must be updated at each time step.
    # 2. Re-sample the particles.
    #    Concept: Now we have the reweighted (unnormalized) distribution, we can now
    #             re-sample the particles from this distribution, choosing a new grid location
    #             for each of the |self.NUM_PARTICLES| new particles. To be extra clear: these
    #             new NUM_PARTICLES should be sampled from the new re-weighted distribution,
    #             not the old belief distribution, with replacement so that more than
    #             one particle can be at a tile
    #
    # - agentX: x location of your car (not the one you are tracking)
    # - agentY: y location of your car (not the one you are tracking)
    # - observedDist: true distance plus a mean-zero Gaussian with standard deviation Const.SONAR_STD
    #
    # Notes:
    # - Remember that |self.particles| is a dictionary with keys in the form of
    #   (row, col) grid locations and values representing the number of particles at
    #   that grid square.
    # - In order to work with the grader, you must create a new dictionary when you are
    #   re-sampling the particles, then set self.particles equal to the new dictionary at the end.
    # - Create |self.NUM_PARTICLES| new particles during resampling.
    # - To pass the grader, you must call util.weightedRandomChoice() once per new
    #   particle.  See util.py for the definition of weightedRandomChoice().
    # - Although the gaussian pdf is symmetric with respect to the mean and value,
    #   you should pass arguments to util.pdf in the correct order
    ##################################################################################
    def observe(self, agentX: int, agentY: int, observedDist: float) -> None:
        # BEGIN_YOUR_CODE
        """
        1. Create two dictionaries called "reweight" and "new_particles".
        2. For each items in self.particles, compute the distance observation and your position.
        3. Use util.pdf() to compute the probability density for a Gaussian where its mean is the distance.
        4. Put the pdf times the number of particles at that grid square in "reweight" which its key is the
           grid location.
        5. Resample |self.NUM_PARTICLES| times using util.weightedRandomChoice() in "new_particles".
        6. Update self.particles to the new_particles.
        """
        reweight = collections.defaultdict(float)
        new_particles = collections.defaultdict(int)
        for (row, col), num in self.particles.items():
            dist = ((util.colToX(col) - agentX) ** 2 + (util.rowToY(row) - agentY) ** 2) ** 0.5
            reweight[(row, col)] = num * util.pdf(dist, Const.SONAR_STD, observedDist)
        for i in range(self.NUM_PARTICLES):
            new_particles[util.weightedRandomChoice(reweight)] += 1
        self.particles = new_particles
        # END_YOUR_CODE

        self.updateBelief()

    ##################################################################################
    # Part 3-2:
    # Function: Elapse Time (propose a new belief distribution based on a learned transition model)
    # ---------------------
    # Reads |self.particles|, representing particle locations at time $t$, and
    # writes an updated |self.particles| with particle locations at time $t+1$.
    #
    # This algorithm takes one step:
    # 1. Proposal based on the particle distribution at current time $t$.
    #    Concept: We have a particle distribution at current time $t$, and we want
    #             to propose the particle distribution at time $t+1$. We would like
    #             to sample again to see where each particle would end up using
    #             the transition model.
    #
    # Notes:
    # - Transition probabilities are stored in |self.transProbDict|.
    # - To pass the grader, you must loop over the particles using a statement
    #   of the form 'for particle in self.particles: <your code>' and call
    #   util.weightedRandomChoice() to sample a new particle location.
    # - Remember that if there are multiple particles at a particular location,
    #   you will need to call util.weightedRandomChoice() once for each of them!
    # - You should NOT call self.updateBelief() at the end of this function.
    ##################################################################################
    def elapseTime(self) -> None:
        # BEGIN_YOUR_CODE
        """
        1. Create a new dictionary called "new_particles".
        2. For every items in self.particles, run "num" times to update the "new_particles" using
           util.weightedRandomChoice() with the transition probabilities of the location.
        3. Update self.particles to the "new_particles".
        """
        new_particles = collections.defaultdict(int)
        for particle, num in self.particles.items():
            for i in range(num):
                new_particles[util.weightedRandomChoice(self.transProbDict[particle])] += 1
        self.particles = new_particles
        # END_YOUR_CODE

    # Function: Get Belief
    # ---------------------
    # Returns your belief of the probability that the car is in each tile. Your
    # belief probabilities should sum to 1.
    def getBelief(self) -> Belief:
        return self.belief
