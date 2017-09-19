I recently had occasion to derive the policy gradient for discrete actions with
a softmax policy, and thought that I would share my solution (in excruciating
detail) as a reference for others who are completing the exercise.

For an explanation of reinforcement learning with policy gradient methods, see
[Lecture 7](http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Teaching_files/pg.pdf)
of David Silver's Reinforcement Learning course.

This will be the derivation for the policy gradient of a discrete action space
policy where the conditional action probability, $$\pi(a|s)$$, is determined by
the softmax of a logit function.

I will use $$\Theta_a$$ to indicate the $$a^{th}$$ row of $$\Theta$$, and
$$\Theta_{ij}$$ to indicate the element in the $$i^{th}$$ row and $$j^{th}$$
column of $$\Theta$$. I will also assume that the state vector, $$s$$, is a
column vector.

$$
\begin{align}
\pi(a|s) &= softmax(\Theta s)_a \\
&= \frac{\exp(\Theta_a s)}{\sum_{a}{\exp(\Theta_a s)}}
\end{align}
$$

# Partial Derivatives
To find $$\nabla_{\Theta} \log \pi(a|s)$$, we can split out each partial
derivative in the gradient matrix:

$$
\begin{align}
\frac{\partial}{\partial \Theta_{ij}} \log \pi(a|s) 
&= \frac{\partial}{\partial \Theta_{ij}} \log\left(\frac{\exp(\Theta_a s)}{\sum_{k}{\exp(\Theta_k s)}}\right) \\
&= \frac{\partial}{\partial \Theta_{ij}} \left(\log(\exp(\Theta_a s)) - \log(\sum_{k}{\exp(\Theta_k s)})\right) \\
&= \frac{\partial}{\partial \Theta_{ij}} \Theta_a s - \frac{\partial}{\partial \Theta_{ij}} \log \sum_{k}{\exp(\Theta_k s)} \\
&= \frac{\partial}{\partial \Theta_{ij}} \sum_{n}{\Theta_{an}s_{n}} - \frac{\partial}{\partial \Theta_{ij}} \log \sum_{k}{\exp(\Theta_k s)}
\end{align}
$$

Expanding the cases for the first term, we have
$$
\begin{align}
\frac{\partial}{\partial \Theta_{ij}} \sum_{n}{\Theta_{an}s_{n}}
&= \begin{cases}
        \frac{\partial}{\partial \Theta_{ij}} \sum_{n}{\Theta_{in} s_{n}} & i = a \\
        0 & i \neq a
    \end{cases} \\
&= \begin{cases}
        \sum_{n} \frac{\partial}{\partial \Theta_{ij}} \Theta_{in} s_{n} & i = a \\
        0 & i \neq a
    \end{cases} \\
&= \begin{cases}
        \{\frac{\partial}{\partial \Theta_{ij}} \Theta_{ij} s_{j}\}_{j=n} + \sum_{j \neq n}{0} & i = a \\
        0 & i \neq a
    \end{cases} \\
&= \begin{cases}
        s_{j} & i = a \\
        0 & i \neq a
    \end{cases}
\end{align}
$$

Continuing with the second term:

Let
$$
\begin{align}
f &= \sum_{k}{\exp(\Theta_k s)}
g_k &= \Theta_{k} s
\end{align}
$$

Then,
$$
\begin{align}
\frac{\partial}{\partial \Theta_{ij}} \log \sum_{k}{\exp(\Theta_{k} s)} &= \frac{\partial}{\partial \Theta_{ij}} \log(f) \\
&= \frac{\partial}{\partial f} \log(f) \frac{\partial}{\Theta_{ij}} f \\
&= \frac{1}{f} \frac{\partial}{\Theta_{ij}} f \\
&= \frac{1}{\sum_{k}{\exp(\Theta_k s)}} \frac{\partial}{\partial \Theta_{ij}} \sum_{k}{\exp(\Theta_{k} s)} \\
&= \frac{1}{\sum_{k}{\exp(\Theta_k s)}} \sum_{k}{\frac{\partial}{\partial \Theta_{ij}} \exp(\Theta_{k} s)} \\
&= \frac{1}{\sum_{k}{\exp(\Theta_k s)}} \sum_{k}{\frac{\partial}{\partial g_k} \exp(g_k) \frac{\partial}{\partial \Theta_{ij}} g_k} \\
&= \frac{1}{\sum_{k}{\exp(\Theta_k s)}} \sum_{k}{\exp(g_k) \frac{\partial}{\partial \Theta_{ij}} g_k} \\
&= \frac{1}{\sum_{k}{\exp(\Theta_k s)}} \sum_{k}{\exp(\Theta_{k} s) \frac{\partial}{\partial \Theta_{ij}} \Theta_{k} s} \\
&= \frac{1}{\sum_{k}{\exp(\Theta_k s)}} \sum_{k}{\exp(\Theta_{k} s) \frac{\partial}{\partial \Theta_{ij}} \sum_{n}{\Theta_{kn} s_{n}}} \\
&= \frac{1}{\sum_{k}{\exp(\Theta_k s)}} \sum_{k}{\exp(\Theta_{k} s) \sum_{n}{\frac{\partial}{\partial \Theta_{ij}} \Theta_{kn} s_{n}}} \\
\frac{\partial}{\partial \Theta_{ij}} \Theta_{kn} s_{n} &= 
    \begin{cases}
        s_{j} & i=k, j=n \\
        0 & otherwise
    \end{cases} \\
&= \frac{1}{\sum_{k}{\exp(\Theta_k s)}} \left(\{\exp(\Theta_{i} s) s_{j}\}_{i=k,j=n} + \sum_{i \neq k}{\sum_{j \neq n}{0}}\right) \\
&= \frac{\exp(\Theta_{i} s)}{\sum_{k}{\exp(\Theta_k s)}} s_{j}
\end{align}
$$

Combining the results for the first and second parts of the equation, we have
the partial derivative solution.

$$
\begin{align}
\frac{\partial}{\partial \Theta_{ij}} \log \pi(a|s)
&= \frac{\partial}{\partial \Theta_{ij}} \sum_{n}{\Theta_{an}s_{n}} - \frac{\partial}{\partial \Theta_{ij}} \log \sum_{k}{\exp(\Theta_k s)} \\
&=
    \begin{cases}
        s_{j} - \frac{\exp(\Theta_{i} s)}{\sum_{k}{\exp(\Theta_k s)}} s_{j} & i = a \\
        0 - \frac{\exp(\Theta_{i} s)}{\sum_{k}{\exp(\Theta_k s)}} s_{j} & i \neq a \\
    \end{cases} \\
&= \begin{cases}
        (1-\pi(a_i | s))s_{j} & i = a \\
        (-\pi(a_i | s))s_{j} & i \neq a
    \end{cases}
\end{align}
$$

# Converting to Matrix Notation
Now that we have the partial derivative solution, we can convert it to the more
compact matrix notation. Let $$e_i$$ represent a one-hot column vector that is
non-zero at index $$i$$, and $$\pi(\cdot|s)$$ represent the column vector of the
probability of each action conditioned on the state.

$$
\begin{align}
\nabla_\Theta \log \pi(a|s) &= (e_a - \pi(\cdot|s))s^T
\end{align}
$$

We can see that the matrix notation matches the partial derivative notation
with the following example where a = 1:

$$
\begin{align}
\nabla_\Theta \log \pi(a|s) &= (e_a - \pi(\cdot|s))s^T \\
&= \left(\left[\begin{array}{c} 0 \\ 1 \\ 0 \end{array}\right] -
\left[\begin{array}{c} \pi(a_0|s) \\ \pi(a_1|s) \\ \pi(a_2|s)
\end{array}\right]\right) \left[\begin{array}{ccc} s_0 & s_1 & s_2
\end{array}\right] \\
&= \left[
\begin{array}{ccc}
-\pi(a_0|s)s_0 & -pi(a_0|s)s_1 & -pi(a_0|s)s_2 \\
(1-\pi(a_1|s))s_0 & (1-pi(a_1|s))s_1 & (1-pi(a_1|s))s_2 \\
-\pi(a_2|s)s_0 & -pi(a_2|s)s_1 & -pi(a_2|s)s_2
\end{array}
\right]
\end{align}
$$

We can easily see that this is the same as the partial derivative solution
above with row $$i$$ and column $$j$$.

For rows where $$i=a=1$$, as defined above for our example, we have
$$
\frac{\partial}{\partial \Theta_{1j}} = (1 - \pi(a_1|s))s_{j}
$$

For rows where $$i \neq a = 1$$, we have
$$
\frac{\partial}{\partial \Theta_{ij}} = (-\pi(a_i|s))s_{j}
$$

