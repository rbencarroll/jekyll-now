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

$$
\begin{align}
\frac{\partial}{\partial \Theta_{ij}} \log \pi(a|s) 
&= \frac{\partial}{\partial \Theta_{ij}} \log\left(\frac{\exp(\Theta_a s)}{\sum_{k}{\exp(\Theta_k s)}}\right) \\
&= \frac{\partial}{\partial \Theta_{ij}} \left(\log(\exp(\Theta_a s)) - \log(\exp(\sum_{k}{\exp(\Theta_k s)}))\right) \\
&= \frac{\partial}{\partial \Theta_{ij}} \Theta_a s - \frac{\partial}{\partial \Theta_{ij}} \log \sum_{k}{\exp(\Theta_k s)}
\end{align}
$$
