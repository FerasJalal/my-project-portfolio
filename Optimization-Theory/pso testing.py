# -*- coding: utf-8 -*-
"""
PSO Testing Version
Used for Activity 3 – Algorithm Testing & Debugging
"""

import random
import numpy
from EvoloPy.solution import solution
import time


def PSO(objf, lb, ub, dim, PopSize, iters):

    # PSO parameters
    Vmax = 6
    wMax = 0.9
    wMin = 0.2
    c1 = 2
    c2 = 2

    s = solution()

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    # ------------------ Initialization ------------------
    vel = numpy.zeros((PopSize, dim))

    pBestScore = numpy.zeros(PopSize)
    pBestScore.fill(float("inf"))

    pBest = numpy.zeros((PopSize, dim))
    gBest = numpy.zeros(dim)
    gBestScore = float("inf")

    pos = numpy.zeros((PopSize, dim))
    for i in range(dim):
        pos[:, i] = numpy.random.uniform(0, 1, PopSize) * (ub[i] - lb[i]) + lb[i]

    convergence_curve = numpy.zeros(iters)

    # ------------------ INITIAL POPULATION ------------------
    print("\n================ INITIAL POPULATION ================")
    for i in range(PopSize):
        print(f"Particle {i} initial position:", pos[i])
        print(f"Particle {i} initial velocity:", vel[i])

    print('\nPSO is optimizing "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    # ================= MAIN LOOP =================
    for l in range(iters):

        print(f"\n================ ITERATION {l+1} ================")

        # -------- Fitness evaluation --------
        for i in range(PopSize):

            for j in range(dim):
                pos[i, j] = numpy.clip(pos[i, j], lb[j], ub[j])

            fitness = objf(pos[i, :])

            print(f"\nParticle {i}")
            print("Position:", pos[i])
            print("Fitness:", fitness)

            if pBestScore[i] > fitness:
                pBestScore[i] = fitness
                pBest[i, :] = pos[i, :].copy()
                print("→ pBest updated for particle", i)

            if gBestScore > fitness:
                gBestScore = fitness
                gBest = pos[i, :].copy()
                print("→ gBest updated")

        # -------- Inertia weight --------
        w = wMax - l * ((wMax - wMin) / iters)
        print("\nInertia weight w:", w)

        # -------- Velocity & Position Update --------
        for i in range(PopSize):
            for j in range(dim):

                r1 = random.random()
                r2 = random.random()

                # 🔍 DEBUG PRINTS FOR DIMENSION 0 ONLY
                if j == 0:
                    print("\n--- Dimension 0 Detailed Update ---")
                    print("Particle:", i)
                    print("x(t):", pos[i, j])
                    print("v(t):", vel[i, j])
                    print("pBest:", pBest[i, j])
                    print("gBest:", gBest[j])
                    print("r1:", r1)
                    print("r2:", r2)

                vel[i, j] = (
                    w * vel[i, j]
                    + c1 * r1 * (pBest[i, j] - pos[i, j])
                    + c2 * r2 * (gBest[j] - pos[i, j])
                )

                if vel[i, j] > Vmax:
                    vel[i, j] = Vmax
                if vel[i, j] < -Vmax:
                    vel[i, j] = -Vmax

                pos[i, j] = pos[i, j] + vel[i, j]

                if j == 0:
                    print("v(t+1):", vel[i, j])
                    print("x(t+1):", pos[i, j])
                    print("---------------------------------")

        convergence_curve[l] = gBestScore

        print("\nIteration summary:")
        print("Global best fitness:", gBestScore)
        print("Global best position:", gBest)

    # ================= END =================
    timerEnd = time.time()

    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence_curve
    s.optimizer = "PSO"
    s.bestIndividual = gBest
    s.objfname = objf.__name__

    return s
