# M/V-ATTA

(Under construction)

## Introduction

This page presents the code for the uncertainty calibration methods M-ATTA and V-ATTA, presented in the paper "Approaching Test Time Augmentation in the Context of Uncertainty Calibration for Deep Neural Networks", submitted to the IEEE Transactions on Pattern Analysis and Machine Intelligence. 

## Method description

In this section we introduce \textit{M-ATTA} and \textit{V-ATTA}. Both methods leverage the use of test time augmentation, combined with a custom adaptive weighting system. \textit{V-ATTA} can be interpreted as restricted version of the more general \textit{M-ATTA}.

\begin{algorithm}[t]
 
    \KwInput{Augmented logits matrix ($\mathbf{Z} \in \mathbb{R}^{k , m}$), Original prediction ($\mathbf{p}^0 \in \Delta_k$)}
 
	\KwOut{Calibrated prediction ($\mathbf{p} \in \Delta_k$)}
    \KwParameters{$\mathbf{W}\in \mathbb{R}^{k , m}$, $\omega^* \in \mathbb{R}$}
  
    $\epsilon \gets 0.01$  %\Comment{Define decreasing step}
     
     $\tilde{\omega} \gets \omega^*$  %\Comment{Initiate }
     
     $c_0 \gets \argmax_{i \in \{1,\ldots,k\}} \mathbf{p}^0$   %\Comment{Get uncalib. prediction class}
     
     
     %\Comment{exit when unclib. and calib. pred. class  match or $\tilde{\omega} \leq 0$}
     
    \While{$c \neq c_0\  \land \ \tilde{\omega} >0$}{ 
            
             
         $\mathbf{p} \leftarrow (1-\tilde{\omega})\mathbf{p}^0 + \tilde{\omega} \text{ } \sigma(\mathbf{W} \odot \mathbf{Z}) I_m$  %\Comment{Update calib. predictions}
         
         $c \gets \argmax_{i \in \{1,\ldots,k\}} \mathbf{p}$ %\Comment{Get  calib. prediction class}
         
         $\tilde{\omega} \gets \tilde{\omega} - \epsilon$  %\Comment{Decrease value}
     }
   \caption{M-ATTA} \label{alg:m-atta}
 \end{algorithm}

\subsection{M-ATTA}
\label{subsect:M-Atta}

Let us start by considering $m \in \mathbb{N}$ different types of augmentations. Because it is common that some augmentations have random parameters, it can be desirable to apply the same type of augmentation more than once; for this reason, let us consider as $n_i$ ($i=1,\ldots,m$) the number of times the $i$-nth type of augmentation is applied. As such, we will define for each original input $I_0$, the $j$-nth ($j=1,\ldots,n_i$) augmented input with the $i$-nth augmentation type, as $I_{i,j}$. \\
\indent We can now take into account the model $f: X \rightarrow \Delta_k$ (where $X$ is the input space and $k$ the number of classes) and consider $g: X \rightarrow \mathbb{R}^k$ as such that $f := g \circ \sigma$ . With this, we now define
\begin{align}
    p^0 = f(I_0), \qquad z_{i,j} = g(I_{i,j}),
\end{align}
\ie, $p^0$ is the probability vector associated with the original input and $z_{i,j}$ is the logit associated with the $j$-nth augmentation of the $i$-nth type.
Subsequently, we can define, $\forall i \in [1,\ldots,m]$, 
\begin{align}
    \mathbf{z}^i = \frac{\sum_{j=1}^{n_i} \mathbf{z}_{i,j}}{n_i}
    \equiv \left(z^i_1, z^i_2, \ldots, z^i_k\right)
    \in \mathbb{R}^k,
\end{align}
and then construct the matrix
\begin{align}
    \mathbf{Z}= \left[ \mathbf{z}^i \right]_{i=1,\ldots,m}^\text{T} =
\left[ \begin{array}{ccccc}
z^1_1 & z^2_1 &  \cdots  & z^m_1 \medskip \\ 
z^1_2 & z^2_2   &  \cdots  & z^m_2   \\
  \vdots         &      \vdots       &     \ddots    & \vdots   \\
  z^1_k         &     z^2_k       &     \cdots    & z^m_k
\end{array}\right] \in \mathbb{R}^{k,m}.
\end{align}
Now, for some parameters 
\begin{align}
\omega^* \in  [0,1], \quad \mathbf{W}=
\left[ \begin{array}{ccccc}
\omega^1_1 & \omega^2_1 &  \cdots  & \omega^m_1 \medskip \\ 
\omega^1_2 & \omega^2_2   &  \cdots  & \omega^m_2   \\
  \vdots         &      \vdots       &     \ddots    & \vdots   \\
  \omega^1_k         &     \omega^2_k       &     \cdots    & \omega^m_k
\end{array}\right] \in \mathbb{R}^{k,m},
\end{align}
we finally define, for each prediction, the final prediction probability vector as
\begin{align}
    \mathbf{p}\left(\tilde{\omega}\right) = 
    (1-\tilde{\omega})\mathbf{p}^0 + \tilde{\omega} \text{ } \sigma(\mathbf{W} \odot \mathbf{Z}) I_m,
\end{align}
with
\begin{align}
\label{omega}
    \tilde{\omega} = \max \Big{\{} \omega \in [0,\omega^*] : \argmax_{i \in \{1,\ldots,k\}} \mathbf{p}\left(\tilde{\omega}\right) = \argmax_{i \in \{1,\ldots,k\}} \mathbf{p}^0 \Big{\}}.
\end{align}
We consider $I_m \in \mathbb{R}^m$ as an $m$ dimensional vector where every element equals 1 and remind that $\sigma: \mathbb{R}^k \rightarrow \Delta_k$ represents the \textit{softmax} function. We also note that the learnable parameters $\mathbf{W}\in \mathbb{R}^{k , m}$  and $\omega^* \in \mathbb{R}$ work, respectively, as an weight matrix and an upper bound for $\tilde{\omega}$. \\
\indent The value of $\tilde{\omega}$ may vary in each prediction, adapting in a way that prevents corruptions in terms of accuracy, according to the definition in \eqref{omega}. Both $\omega^* \in [0,1]$ and $\mathbf{W} \in \mathbb{R}^{k,m}$ can be optimized with a given validation set. In a practical scenario, the value $\tilde{\omega}$ is approximated as described in the algorithmic description of \textit{M-ATTA} in Algorithm \ref{alg:m-atta}. In our case $\epsilon$ in Algorithm \ref{alg:m-atta} is defined as 0.01. 




\subsection{V-ATTA}
\label{subsection:V-Atta}

With \textit{V-ATTA} we restrict the matrix $\mathbf{W}$ to a diagonal matrix
\begin{align}
    \mathbf{W}_d=
\left[ \begin{array}{ccccc}
\omega^1 & 0 &  \cdots  & 0 \medskip \\ 
0 & \omega^2   &  \cdots  & 0  \\
  \vdots         &      \vdots       &     \ddots    & \vdots   \\
  0        &     0       &     \cdots    & \omega^m
\end{array}\right] \in \mathbb{R}^{m,m},
\end{align}
and define the new prediction probability vector as
\begin{align}
    \mathbf{p}\left(\tilde{\omega}\right) = 
    (1-\tilde{\omega})\mathbf{p}^0 + \tilde{\omega} \text{ } \sigma (\mathbf{W}_d  \mathbf{Z}^{\text{T}}) I_m,
\end{align}
with $\tilde{\omega}$ still defined as in \eqref{omega}.\\
\indent We care to note that, contrarly to \textit{M-ATTA}, \textit{V-ATTA} has the same number of parameters as the preliminary method presented in \cite{Conde_2022_BMVC}, while still being able of accomplishing improved results (see Supplementary Material, Section 3). \\
\indent In this case, the algorithmic description is as represented in Algorithm \ref{alg:v-atta}. Once again, $\epsilon$ is 0.01.



\begin{algorithm}[h]

    
    \KwInput{Augmented logits matrix ($\mathbf{Z} \in \mathbb{R}^{k , m}$), Original prediction ($\mathbf{p}^0 \in \Delta_k$)}
 
	\KwOut{Calibrated prediction ($\mathbf{p} \in \Delta_k$)}
    \KwParameters{$\mathbf{W}_d \in \mathbb{R}^{m , m}$, $\omega^* \in \mathbb{R}$}
  
    $\epsilon \gets 0.01$  %\Comment{Define decreasing step}
     
     $\tilde{\omega} \gets \omega^*$  %\Comment{Initiate }
     
     $c_0 \gets \argmax_{i \in \{1,\ldots,k\}} \mathbf{p}^0$   %\Comment{Get uncalib. prediction class}
     
     
     %\Comment{exit when unclib. and calib. pred. class  match or $\tilde{\omega} \leq 0$}
     
    \While{$c \neq c_0\  \land \ \tilde{\omega} >0$}{ 
            
             
         $\mathbf{p} \leftarrow (1-\tilde{\omega})\mathbf{p}^0 + \tilde{\omega} \text{ } \sigma (\mathbf{W}_d  \mathbf{Z}^{\text{T}}) I_m$  %\Comment{Update calib. predictions}
         
         $c \gets \argmax_{i \in \{1,\ldots,k\}} \mathbf{p}$ %\Comment{Get  calib. prediction class}
         
         $\tilde{\omega} \gets \tilde{\omega} - \epsilon$  %\Comment{Decrease value}
     }
   \caption{V-ATTA} \label{alg:v-atta}
 \end{algorithm}
