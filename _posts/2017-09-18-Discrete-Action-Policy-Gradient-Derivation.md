I recently had occasion to derive the discrete action policy gradient, and
thought that I would share my solution as a reference for others who were
completing the exercise.

$$
\begin{align}
\pi(a|s) &= softmax(\Theta s)_a 
&= \frac{exp(\Theta_a^T s)}{\sum_{a}{exp(\Theta_a^T s)}}
\end{align}
$$

$$ \frac{\partial}{\partial \Theta_{ij}} log \pi(a|s)$$
