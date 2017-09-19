I recently had occasion to derive the discrete action policy gradient, and
thought that I would share my solution as a reference for others who were
completing the exercise.

$$
\begin{align}
\pi(a|s) &= softmax(\Theta s)_a \\
&= \frac{\exp(\Theta_a s)}{\sum_{a}{\exp(\Theta_a s)}}
\end{align}
$$

To find $$\nabla_{\Theta} \log \pi(a|s)$$, we can split out each component of
the gradient matrix:

I will use $$\Theta_a$$ to indicate the $$a^{th}$$ row of $$\Theta$$, and
$$\Theta_{ij}$$ to indicate the element in the $$i^{th}$$ row and $$j^{th}$$
column of $$\Theta$$. I will also assume that the state vector, $$s$$, is a
column vector.

$$
\begin{align}
\frac{\partial}{\partial \Theta_{ij}} \log \pi(a|s) 
&= \frac{\partial}{\partial \Theta_{ij}} \log\left(\frac{\exp(\Theta_a s)}{\sum_{k}{\exp(\Theta_k s)}}\right) \\
&= \frac{\partial}{\partial \Theta_{ij}} \left(\log(\exp(\Theta_a s)) - \log(\exp(\sum_{k}{\exp(\Theta_k s)}))\right) \\
&= \frac{\partial}{\partial \Theta_{ij}} \Theta_a s - \frac{\partial}{\partial \Theta_{ij}} \log \sum_{k}{\exp(\Theta_k s)} \\
&= \frac{\partial}{\partial \Theta_{ij}} \sum_{n}{\Theta_{an}s_{n}} - \frac{\partial}{\partial \Theta_{ij}} \log \sum_{k}{\exp(\Theta_k s)}
\end{align}
$$

Expanding the cases for the first term, we have
$$
\begin{align}
\frac{\partial}{\partial \Theta_{ij}} \sum_{n}{\Theta_{an}s_{n}} &=
\begin{cases}
\frac{\partial}{\partial \Theta_{ij}} \sum_{n}{\Theta_{in} s_{n}} & i = a, j = n \\
0 & i \neq a
\end{cases} \\
&= \begin{cases}
\frac{\partial}{\partial \Theta_{ij}} \Theta_{ij} s_{j} & i = a, j = n \\
0 & i \neq a
\end{cases} \\
&= \begin{cases}
s_{j} & i = a, j = n \\
0 & i \neq a
\end{cases}
\end{align}
$$
