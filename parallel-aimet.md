# End-to-End AIMET Instance Parallelization

This document formalizes the AIMET sensitivity scan parallelization pipeline:
the mathematical cost model, the step-by-step orchestration flow, and the
resulting speedup and efficiency metrics.

---

## Equation (1): AIMET Quantizer Sweep Pipeline

The sweep evaluates each quantizer in isolation and collects the results into
a single JSON artifact $\mathcal{J}$:

$$
\mathcal{J}
=
\Phi
\Big(
\big\{
\big(
i,\;
\mathcal{E}
\big(
M(\mathbf{z}^{(i)}),\;
\mathcal{D}
\big)
\big)
\;\big|\;
i \in \{0,\dots,QL-1\}
\big\}
\Big)
\qquad
\left\{
\begin{array}{ll}
M & \text{: base model} \\
\mathcal{Q} = \{Q_0,\dots,Q_{QL-1}\} & \text{: quantizer set} \\
QL & \text{: total quantizers} \\
i & \text{: quantizer index} \\
\mathbf{z}^{(i)} \in \{0,1\}^{QL} & \text{: one-hot enable vector} \\
\mathcal{D} & \text{: evaluation dataset} \\
\mathcal{E}(M,\mathcal{D}) \to \mathbb{R}^K & \text{: evaluation function (SQNR)} \\
\Phi(\cdot) & \text{: serialization function} \\
\mathcal{J} & \text{: output JSON}
\end{array}
\right.
$$

where the one-hot enable vector is defined as:

$$
z^{(i)}_j =
\begin{cases}
1 & j = i \\
0 & j \neq i
\end{cases}
$$

---

## Equation (2): Parallelized AIMET Runtime Model

$$
T_{\text{E2E}}(\mathcal{J})
=
\frac{QL \cdot W}{P}
=
\frac{QL \cdot W}{
  K \cdot \left\lfloor \dfrac{G + \text{safety}_{\text{net}}}{U} \right\rfloor
}
=
\frac{QL \cdot W}{
  K \cdot \left\lfloor \dfrac{G + \text{safety}_{\text{net}}}{u_m + u_d} \right\rfloor
}
\qquad
\left\{
\begin{array}{ll}
u_m & \text{: VRAM per model instance} \\
u_d & \text{: VRAM per dataloader} \\
U = u_m + u_d & \text{: total VRAM per instance} \\
G & \text{: available VRAM per GPU} \\
\text{safety}_{\text{net}} & \text{: reserved VRAM buffer} \\
K & \text{: number of GPUs} \\
P & \text{: total parallel instances} \\
W & \text{: time per evaluation (min)} \\
T_{\text{serial}} = QL \cdot W & \text{: serial runtime} \\
T_{\text{parallel}} = QL \cdot W / P & \text{: parallel wall-clock runtime}
\end{array}
\right.
$$

### Per-Inference Memory Model

$$
U = u_m + u_d
$$

### Total Parallel Instances

$$
P =
K \cdot
\left\lfloor
\dfrac{G + \text{safety}_{\text{net}}}{U}
\right\rfloor
$$

### Parallel Runtime

$$
T_{\text{parallel}}
=
\frac{QL \cdot W}{P}
$$

### Baseline (Serial Runtime)

$$
T_{\text{serial}} = QL \cdot W
$$

---

## Collapsed Cost Equation

$$
T_{\text{E2E}}(\mathcal{J})
=
\frac{
\left|
\big\{
i \;\big|\; i \in \{0,\dots,QL-1\}
\big\}
\right|
\cdot W
}{
K \cdot
\left\lfloor
\dfrac{G + \text{safety}_{\text{net}}}{u_m + u_d}
\right\rfloor
}
$$

Since $\left|\{ i \mid i \in \{0,\dots,QL-1\} \}\right| = QL$, this reduces to

$$
T_{\text{E2E}}
=
\frac{QL \cdot W}{P}
\qquad \text{where} \qquad
P = K \cdot \left\lfloor \dfrac{G + \text{safety}_{\text{net}}}{U} \right\rfloor
$$

---

## Speedup and Efficiency Metrics

### Speedup

$$
S
=
\frac{T_{\text{serial}}}{T_{\text{E2E}}}
=
K \cdot
\left\lfloor
\dfrac{G + \text{safety}_{\text{net}}}{U}
\right\rfloor
=
P
$$

**Interpretation:**  
Speedup equals the total number of concurrent AIMET evaluation instances.

### Parallel Efficiency

$$
\eta
=
\frac{S}{K}
=
\left\lfloor
\dfrac{G + \text{safety}_{\text{net}}}{U}
\right\rfloor
$$

**Interpretation:**  
Parallel efficiency measures how well each GPU is utilized by packed
AIMET inference instances.


---

## Key Insight

> AIMET quantizer sweep runtime scales linearly with the number of
> quantizers and inversely with the GPU memory–bounded parallel capacity,
> enabling order-of-magnitude reductions in wall-clock time through
> instance-level parallelism.
