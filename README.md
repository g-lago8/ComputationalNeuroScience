# CNS assignments

## Assignment 1
### Izhikevic Model
Implement the Izhikevic model of spiking neurons. The model is described by the following differential equations:
```math
\begin{cases}\dfrac{du}{dt} = 0.04u^2 + 5u + 140 - w + I \\

\dfrac{dw}{dt} = a(bu - w)  \end{cases}
```



where 
- $u$ is the membrane potential,
- $w$ is the recovery variable,
- $a$, $b$, $c$, and $d\in \mathbb{R}$ are parameters that define the behavior of the neuron,
- $I:\mathbb{R}^{+}\to \mathbb{R}$ is the input current.

Show the response of the neuron to different input currents, in particular, there are 20 different computational features of the neuron that have to be shown.

### Bonus 1
Implement the Liquid State Machine and use it for a task of autoregression on the sun spots dataset.

A Liquid State Machine (LSM) is a type of recurrent neural netwoek that uses a pool of spiking neurons as a reservoir. The reservoir is driven by the input signal and the output is obtained by training a readout layer on the reservoir states. The reservoir is left untrained, and made of a percentage of excitatory and inhibitory neurons.

### Bonus 2
Train the LSM reservoir using a simplified version of the STDP (Spike Time Dependent Plasticity) rule.

## Assignment 2

### Assignment 2.1: Hebbian Learning
Train a simple single-layer network of neurons with the Hebbian Learning paradigm and its variants.

Hebbian Learning is an unsupervised learning paradigm that is shown in synapses of real neurons.

The basic hebbian learning update rule is $w \leftarrow  w +\eta u^Tv$,  where $u$ is the real input and $v$ is the prediction $v=uw^T$.

There are a lot of other learning rules that fix some problems of vanilla Hebbian rule, in particular the fact that the dynamical system associated with the update is not stable, thus $\| w^{(t)}\|\to \infty$.

In this assignment are implemented

- basic Hebbian update
- Oja rule
- Subtractive Normalization
-
