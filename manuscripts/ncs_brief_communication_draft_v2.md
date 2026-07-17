## Abstract

Protein language models (pLMs) increasingly dominate sequence-based protein–protein interaction (PPI) prediction, yet the biological mechanisms driving their reported accuracy remain opaque. Because benchmark performance can arise from network-topology shortcuts, dataset leakage, or biased negative sampling rather than physical binding, accuracy alone cannot certify that a predictor has learned interaction mechanisms. Here we introduce AuditPPI, a post-hoc framework that uses sparse autoencoder (SAE) features of a frozen pLM as an interpretable vocabulary to separate these competing explanations across protein, pair and residue scales. We find that SAE features readily encode protein "hub-ness" and pairwise semantic concordance—signals sufficient to inflate even strict, leakage-controlled benchmarks—but do not, by default, encode contact-specific interface compatibility: enrichment of top benchmark features for true interface residues vanishes under a surface-matched control. Genuine residue-level contact signals do exist within pLM representations, but they are concentrated in a minority of structural classes such as transmembrane and ribosomal contacts rather than distributed across the proteome. AuditPPI reframes PPI evaluation around mechanistic fidelity, providing a diagnostic layer to ensure that future datasets and models capture biophysical interaction principles rather than confounded correlates of accuracy.

## Main

Protein–protein interactions (PPIs) form the operational backbone of cellular machinery, and their accurate computational inference is a cornerstone of modern molecular and structural biology<sup>1</sup>. Two paradigms now dominate. Structure- and coevolution-grounded methods, exemplified by RoseTTAFold2-PPI, exploit paired multiple-sequence alignments to model physical interfaces directly and have enabled proteome-scale interaction screening in human<sup>2</sup>. In parallel, pretrained protein language models (pLMs) have made single-sequence prediction attractive by removing the MSA bottleneck. Early pLM-based predictors embed each protein independently and combine the two fixed representations<sup>3</sup>, whereas a more recent generation jointly encodes the protein pair to capture inter-protein context: PLM-interact adapts a next-sentence-prediction objective, MINT and PPLM introduce cross-protein attention over paired sequences, and grammar-based interaction language models such as SWING tokenize the pair directly<sup>4–7</sup>. These models report strong performance on binding-affinity, mutational-effect, interface-contact and complex-assembly tasks<sup>4,6</sup>. However, for sequence-only predictors the dominant binary framing—classifying pairs as interacting or not—conflates a physical question with a statistical one, and obscures which biological mechanism a model has actually learned.

Despite frequently reporting near-perfect benchmark accuracies, the field faces a systemic crisis of trust driven by data leakage and shortcut learning<sup>8–10</sup>. The core diagnosis dates to a foundational critique of pair-input evaluation, which showed that random pair splits leak information because proteins recur across train and test<sup>8</sup>; this motivated the strict C1–C3 split scheme, in which test pairs share two, one or zero proteins with the training set. Rigorous re-evaluations in the pLM era confirm that exceptional performance is often an algorithmic illusion: accuracy collapses toward chance on the leakage-controlled C3 regime<sup>9</sup>. Rather than learning universal biochemical rules, predictors exploit network-topology shortcuts—memorizing high-degree "hub" proteins—or rely on sequence similarity to copy training labels onto homologous test proteins<sup>9,10</sup>. Artificial negative-sampling strategies compound the problem: pairing proteins from different subcellular compartments inadvertently reduces a PPI predictor to a localization classifier, because pLMs encode subcellular localization with high fidelity<sup>11</sup>, so models fail on hard negatives that co-localize<sup>12</sup>. Network-level benchmarks such as PRING were introduced precisely to penalize this opportunism by demanding cross-species and function-oriented generalization<sup>13</sup>.

Consequently, when a sequence-only model assigns a high score to a protein pair, it can do so for mechanistically distinct and often confounding reasons. It might recognize that one protein has a high global participation rate (the hub shortcut); it might detect semantic concordance—shared cellular context or co-localization (the negative-sampling bias); or it might identify genuine interface grounding, where the two surfaces present compatible binding sites. Only the last reflects a physical interaction mechanism, yet all three inflate pairwise benchmark accuracy, and conventional metrics cannot tell them apart.

Resolving this ambiguity has been hampered by the black-box nature of pLM-based predictors. The recent adaptation of sparse autoencoders (SAEs) to pLMs offers a way in: SAEs decompose dense, polysemantic pLM activations into a large dictionary of sparse, monosemantic features that align with interpretable biological concepts such as binding sites, structural motifs and functional families<sup>14,15</sup>. Leveraging this, we introduce AuditPPI, a framework that treats SAE features not as a new predictor but as a shared, interpretable vocabulary in which to test what drives PPI predictions. We structure the audit around three hypotheses that map directly onto the three confounded signals: (1) individual-protein features encode participation (hub-ness); (2) pairwise feature concordance encodes shared biological context; and (3) residue-level feature co-activation encodes true structural interface compatibility. By separating features that are merely predictive from those that are physically mechanistic, AuditPPI provides a diagnostic layer to ensure that future dataset design and model evaluation prioritize biophysical mechanism over illusory accuracy.

## Results

**A participation signal is recoverable from sequence alone.** A pairwise model can appear accurate simply by recognizing that one partner is a promiscuous hub, since high-degree proteins are over-represented among positives and rarely sampled as negatives<sup>9,10</sup>. We therefore first asked whether protein participation—node degree in the interactome—is itself predictable from sequence. Using the PRING human reference graph<sup>13</sup>, we labelled high-participation proteins as those with a full-graph degree at or above the 90th percentile (degree ≥ 61; 1,021 of 10,090 proteins). An XGBoost<sup>16</sup> classifier trained only on binary SAE features, under the leakage-controlled PRING BFS protein split, predicted this label with a test AUROC of 0.701 and AUPRC of 0.561 (baseline positive rate 0.284). Sequence does not fully determine degree, but it carries a substantial, recoverable participation signal—enough for a pairwise predictor to exploit generic "hub-ness" without modelling any partner-specific complementarity.

**Pairwise feature concordance inflates a strict, leakage-controlled benchmark.** We next examined the pair level under the stringent C3 split, where test pairs share no proteins with training<sup>9</sup>. A classifier on the top 200 ranked SAE features reached a C3 test AUROC of 0.930. Critically, this did not require a trained decision boundary at all: model-free similarity between the two proteins' SAE features separated positive from negative C3 pairs on its own (dense cosine AUROC = 0.717; binary Jaccard AUROC = 0.669). That shared sparse-feature context is intrinsically predictive—absent any explicit interface model—shows how semantic concordance, the signature of localization- and context-based negative sampling<sup>11,12</sup>, can inflate even a leakage-controlled benchmark.

**The C3 signal is partly a retrieval shortcut.** To test whether this concordance acts as a local nearest-neighbour shortcut, we audited a trained TabPFN<sup>17</sup> classifier by reconstructing its decoder attention from each of 4,868 C3 test pairs to all 39,230 training pairs. Most test queries attended to highly label-concordant neighbourhoods (median same-label attention share = 0.862) despite only moderate feature overlap, indicating systematic reliance on labelled neighbours. A high-risk subset (1.09% of test pairs) depended on near-duplicate training examples; in the most extreme case, all 20 top attention neighbours were positive training pairs, capturing 96.2% of the attention mass at a maximum feature Jaccard of 0.928. C3 performance is thus partly supported by feature-similar, label-concordant training neighbours—an advanced retrieval shortcut—rather than de novo interface rules.

**Benchmark-important features are not interface-specific.** We then asked whether the features driving benchmark accuracy reflect genuine physical binding, framing interface grounding as a structural-enrichment problem. From 190,389 non-homologous positive pairs in PDB (PDB_PPI), we defined interface residues by an 8 Å Cβ–Cβ threshold. Against all same-chain non-interface residues, 11.1% of SAE features were significantly interface-enriched (FDR < 0.05, log odds ratio > 0.5), and the top 50 C3 features overlapped this set 1.98-fold (22.0%; P = 0.0195)—an apparent confirmation of mechanistic grounding. This permissive background, however, conflates true contacts with mere surface exposure. We therefore repeated the test against a surface-matched control: solvent-exposed, non-interface residues from the same chains. Under this control the enrichment vanished—only 7 of the top 50 C3 features were interface-specific (1.12-fold; P = 0.435), with null results for the top 100 and 200. Highly predictive C3 features therefore encode broad surface propensity or biological context, not partner-specific interface compatibility, and high benchmark accuracy should not be read as mechanistic fidelity.

**Genuine contact signals exist but are structurally localized.** Finally, to test whether any true binding signal exists in the representations, we measured contact-specific feature-pair compatibility: co-activation of top SAE features at cross-chain contact residue pairs (Cβ–Cβ ≤ 8 Å) versus spatially distant control pairs (> 12 Å). Here 6.6% of feature pairs were significantly contact-compatible, confirming that residue-level interface information is present. But these signals were not proteome-wide; they concentrated in specific structural regimes—transmembrane helix contacts, ribosomal complexes and enzyme active-site pockets—consistent with co-evolutionary contact signals being strongest in tightly packed, evolutionarily constrained interfaces<sup>19</sup>. Authentic contact information is thus real but localized, and cannot account for the broad benchmark success of sequence-only PPI models.

Together, these three audits show that sequence-only PPI accuracy is a conflated metric: a single score blends protein-level participation bias, pair-level semantic concordance and only sparse, structurally localized interface grounding (Fig. 1). The same SAE vocabulary resolves this mixture consistently across the protein, pair and residue scales, isolating the fraction of predictive signal that is physically mechanistic from the larger fraction that is contextual.

## Discussion

Our findings reframe a now-familiar observation—that leakage-controlled splits such as C3 expose inflated PPI accuracy<sup>9,10,13</sup>—by identifying *what* the surviving signal is rather than only *that* it survives. Even under the strict C3 regime, an SAE-feature model reaches AUROC 0.930, but AuditPPI attributes this performance to participation and semantic concordance, plus a nearest-neighbor retrieval component, rather than to learned interface complementarity. This explains why models that never encode partner-specific binding can nonetheless top sequence-only benchmarks, and why such models generalize poorly to network- and function-oriented tasks<sup>13</sup>.

The result also clarifies what SAE features do and do not capture. Consistent with their emerging use as interpretable descriptors of structure, function and localization<sup>14,15,20</sup>, SAE features faithfully encode the contextual properties—hub-ness, compartment, family—that confound PPI labels. Genuine residue-level contact compatibility is present, but concentrated in tightly packed, evolutionarily constrained regimes such as transmembrane and ribosomal contacts where co-evolutionary signal is strongest<sup>19</sup>. This mirrors the design of the most accurate proteome-scale predictors: structure-grounded methods such as RoseTTAFold2-PPI achieve genome-wide screening precisely by exploiting paired-MSA coevolution to model physical interfaces<sup>2</sup>, and recent retrieval- and contact-grounded models succeed by predicting residue-level interfaces directly<sup>17</sup>. Together these suggest that proteome-wide mechanistic prediction will require supervision or architectures that target interface compatibility rather than the global context that sequence-only pLM features most readily expose.

AuditPPI is diagnostic rather than corrective: it flags retrieval-style risk and quantifies mechanistic grounding, but does not by itself remove leakage or prove the causal basis of any single prediction. It is also bounded by the SAE dictionary and by the structural coverage of PDB-derived interfaces, which under-represents transient and disordered complexes. Nonetheless, by separating features that are merely predictive from those that are physically mechanistic, AuditPPI provides a practical layer for benchmark and model design: future datasets should report not only leakage-controlled accuracy but the degree to which that accuracy rests on genuine interface signal.

## Methods

AuditPPI is a post-hoc interpretability framework for decomposing the signals available to sequence-based PPI predictors. Rather than treating terminal classification accuracy as a single biological quantity, the framework re-examines existing benchmark splits, cached protein representations and trained-model artifacts such as feature rankings or TabPFN attention neighbourhoods. Labels are used as the original benchmark annotations for post-hoc comparison, and in the PRING analysis to define graph-derived participation labels; they are not used to define a new PPI scoring model. We organize the analysis around three mechanistic scales: protein-level participation, pair-level sparse-feature concordance and residue-level structural grounding. The framework therefore reports diagnostic comparisons and controls, rather than a single formal audit statistic or a new interaction score.

The pipeline has two computational phases. In phase 1, sparse representation decomposition, each protein sequence is passed through a pretrained protein language model (ESM-C<sup>21</sup>) and a fixed sparse autoencoder trained on its residue embeddings<sup>14,15</sup>. Operationally, for a protein $p$ with length $L_p$, the pLM produces contextual embeddings

$$E_p \in \mathbb{R}^{L_p \times d_{\mathrm{pLM}}}$$

and the fixed SAE encoder maps these embeddings to residue-level sparse activations

$$H_p = \mathrm{SAE}_{\mathrm{enc}}(E_p) \in \mathbb{R}^{L_p \times m}, \qquad m = 16384.$$

The SAE itself was not trained as part of the AuditPPI analyses; it was used as a frozen feature dictionary. Residue-level caches stored the top 64 active SAE feature indices and activation values for each residue. For protein-level analyses we used a continuous max-pooled fingerprint

$$g_{p,k} = \max_{1 \le r \le L_p} H_{p,r,k}$$

and a binary presence fingerprint

$$b_{p,k} = \mathbb{1}\!\left[g_{p,k} > 0\right].$$

Thus, the same feature vocabulary could be queried at protein, pair, residue and residue-pair resolution.

In phase 2, multi-level signal auditing, the sparse fingerprints were tested against three classes of explanations. The first audit quantified generic protein participation. In a reference interaction graph, the empirical participation tendency of protein $p$ was

$$t(p) = \frac{\sum_i \mathbb{1}\!\left[p \in (a_i, b_i)\right]\, y_i}{\sum_i \mathbb{1}\!\left[p \in (a_i, b_i)\right]}.$$

To test whether participation is sequence-predictable, proteins were labelled as high-participation if their graph degree exceeded a fixed quantile threshold,

$$h_p = \mathbb{1}\!\left[\deg(p) \ge Q_{0.9}(\deg)\right],$$

and an audit classifier $F_\theta$, implemented here with XGBoost for the PRING experiments, was trained to estimate

$$P(h_p = 1 \mid g_p) \approx F_\theta(g_p)$$

or the corresponding probability from the binary fingerprint $b_p$. For pair-level benchmark diagnostics we also used a leave-one-out participation oracle. For pair $i$, the protein-level rate excluding the scored pair was

$$t_{-i}(p) = \frac{\sum_{j \ne i} \mathbb{1}\!\left[p \in (a_j, b_j)\right]\, y_j}{\sum_{j \ne i} \mathbb{1}\!\left[p \in (a_j, b_j)\right]},$$

with proteins observed only in the scored pair assigned the global mean of $t(p)$. The pair oracle score was

$$s_{\mathrm{part}}(a_i, b_i) = \min\!\left(t_{-i}(a_i),\, t_{-i}(b_i)\right).$$

Model or fingerprint-pair performance was interpreted relative to this oracle by reporting

$$\Delta_{\mathrm{AUROC}} = \mathrm{AUROC}(s_{\mathrm{model}}, y) - \mathrm{AUROC}(s_{\mathrm{part}}, y),$$

with an analogous AUPRC lift for imbalanced benchmarks.

The second audit measured pair-level concordance and retrieval-neighbour support. This required converting two protein-level SAE fingerprints into a single order-invariant pair fingerprint. Let $x_p$ denote the per-protein representation used for a given experiment: $x_p = b_p$ for the binary SAE fingerprint, $x_p = g_p$ for the continuous max-pooled SAE fingerprint, and an ESM-C mean vector for the non-SAE embedding control. For a selected feature set $S$, the pair fingerprint was the concatenation of two symmetric blocks,

$$\Phi_S(a, b) = \left[\, x_{a,S} \circ x_{b,S},\; \left|x_{a,S} - x_{b,S}\right| \,\right].$$

The first block records shared feature activation, equivalent to an AND state for binary fingerprints; the second records feature discordance, equivalent to an XOR state for binary fingerprints. This construction is invariant to swapping the two proteins in a pair and preserves whether a classifier is using shared sparse semantics or mismatch patterns. For the C3 TabPFN attention audit, $S$ contained the top 200 SAE feature identifiers from the binary symmetric feature ranking, and each identifier contributed both an AND and an XOR column; the resulting TabPFN input therefore had $2|S| = 400$ pair-feature dimensions. In the baseline experiments, classifiers were trained either on the full symmetric matrix $\left[A \circ B,\, |A - B|\right]$ or on XGBoost-ranked top columns when a compact TabPFN input was required.

As model-free concordance controls, we computed dense cosine similarity on the continuous protein fingerprint,

$$\cos(g_a, g_b) = \frac{g_a^{\top} g_b}{\lVert g_a \rVert_2\, \lVert g_b \rVert_2},$$

and binary Jaccard similarity on active SAE feature identities,

$$J(b_a, b_b) = \frac{\lvert A_a \cap A_b \rvert}{\lvert A_a \cup A_b \rvert}, \qquad A_p = \{\, k : b_{p,k} = 1 \,\}.$$

For retrieval-neighbour analysis, we write $\Phi_i = \Phi_S(a_i, b_i)$. Jaccard overlap was computed on the active columns of this selected pair fingerprint, that is, columns with positive AND or XOR values. Thus, a high overlap meant that two pair examples shared the same reduced SAE pair states used by the downstream classifier, not merely that their individual proteins shared many pooled features. For the TabPFN C3 audit, we further reconstructed decoder attention from a test pair $i$ to each training pair $j$,

$$\alpha_{ij} = \operatorname{softmax}_j\!\left(\frac{q_i^{\top} k_j}{\sqrt{d_h}}\right),$$

and retained the top $K = 20$ attention neighbours $N_K(i)$. The attention-weighted feature overlap and label-concordant attention share were

$$J_{w,i} = \frac{\sum_{j \in N_K(i)} \alpha_{ij}\, J(\Phi_i, \Phi_j)}{\sum_{j \in N_K(i)} \alpha_{ij}},$$

$$L_i = \frac{\sum_{j \in N_K(i)} \alpha_{ij}\, \mathbb{1}\!\left[y_i = y_j\right]}{\sum_{j \in N_K(i)} \alpha_{ij}}.$$

The retrieval-risk score was

$$R_i = L_i \cdot \max\!\left\{\, J(\Phi_i, \Phi_j) : j \in N_K(i) \,\right\}.$$

Large values of $R_i$, together with high maximum Jaccard and high same-label neighbour rate, were interpreted as evidence that a prediction may be supported by feature-near labelled examples. This audit flags retrieval-style risk; it does not by itself prove exact duplicate leakage or establish the causal mechanism of a prediction.

The third audit tested structural grounding of SAE features. Using positive PDB_PPI complexes, $C_\beta$-like coordinates were reconstructed for each residue. For a structural pair $(a, b)$, residue $r$ in chain $a$ was labelled as an interface residue when any partner residue was within the contact threshold $\delta = 8\,\text{Å}$:

$$I_{a,r} = \mathbb{1}\!\left[\min\!\left\{\, \lVert C_\beta(a, r) - C_\beta(b, s) \rVert_2 : s \in \mathrm{chain}(b) \,\right\} \le \delta\right].$$

For each SAE feature $k$, active and inactive residues were counted inside the interface and in a control set. The broad control used all same-chain non-interface residues, whereas the stricter control used solvent-exposed non-interface residues from the same chain side. With 0.5 pseudocounts, the interface-enrichment statistic was

$$\log \mathrm{OR}_k = \log \frac{(n_{11} + 0.5)(n_{00} + 0.5)}{(n_{10} + 0.5)(n_{01} + 0.5)},$$

where $n_{11}$ and $n_{10}$ are active and inactive interface counts, and $n_{01}$ and $n_{00}$ are active and inactive control counts. One-sided Fisher exact tests were corrected by the Benjamini–Hochberg procedure. A feature was called interface-grounded only after passing the corrected $q$-value, minimum log-odds and minimum-support thresholds. To test whether benchmark-important features were enriched for structurally grounded features, ranked feature sets were evaluated with a hypergeometric test $X \sim \mathrm{Hypergeom}(N, G, K)$, where $N$ is the SAE vocabulary size, $G$ is the number of interface-grounded features and $K$ is the ranked-set size.

Finally, AuditPPI tested contact-specific feature-pair compatibility, a stricter residue-pair analysis than single-feature interface enrichment. Cross-chain residue pairs with $C_\beta$–$C_\beta$ distance below $8\,\text{Å}$ formed the contact set $C$; surface non-interface residue pairs from the same complex with distance above $12\,\text{Å}$ formed the control set $U$. For each residue, only the top $M = 4$ active SAE features were used. For a cross-chain feature pair $(u, v)$, co-activation counts in contact and control residue pairs were compared by

$$\log \mathrm{OR}_{uv} = \log \frac{(c_C + 0.5)(N_U - c_U + 0.5)}{(c_U + 0.5)(N_C - c_C + 0.5)}.$$

Feature pairs were called contact-compatible only when contact co-activation exceeded control co-activation after multiple-testing correction and minimum-support filtering. Together, the three audit levels separate participation and semantic concordance from residue-level evidence that is closer to physical interface or contact compatibility.

## Data availability

All datasets used in this paper are publicly available.

## Code availability

The source code used in this study is available via GitHub at *****.

## References

1. Snider, J. et al. Fundamentals of protein interaction network mapping. *Mol. Syst. Biol.* **11**, 848 (2015).
2. Zhang, J., Humphreys, I. R., Pei, J. et al. Predicting protein–protein interactions in the human proteome. *Science* **390**, eadt1630 (2025).
3. Wang, R. et al. Nanobody–antigen interaction prediction with ensemble deep learning and prompt-based protein language models. *Nat. Mach. Intell.* **6**, 1594–1607 (2024).
4. Liu, D. et al. PLM-interact: extending protein language models to predict protein–protein interactions. *bioRxiv* https://doi.org/10.1101/2024.11.05.622169 (2024).
5. Ullanat, V., Jing, B., Sledzieski, S. & Berger, B. Learning the language of protein–protein interactions. *bioRxiv* https://doi.org/10.1101/2025.03.09.642188 (2025).
6. Liu, J., Chen, H. & Zhang, Y. A corporative language model for protein–protein interaction, binding affinity, and interface contact prediction. *bioRxiv* https://doi.org/10.1101/2025.07.07.663595 (2025).
7. Siwek, J. C. et al. Sliding Window Interaction Grammar (SWING): a generalized interaction language model for peptide and protein interactions. *Nat. Methods* (2025).
8. Park, Y. & Marcotte, E. M. Flaws in evaluation schemes for pair-input computational predictions. *Nat. Methods* **9**, 1134–1136 (2012).
9. Bernett, J., Blumenthal, D. B. & List, M. Cracking the black box of deep sequence-based protein–protein interaction prediction. *Brief. Bioinform.* **25**, bbae076 (2024).
10. Szymborski, J. & Emad, A. A flaw in using pre-trained pLLMs in protein–protein interaction inference models. *bioRxiv* https://doi.org/10.1101/2025.04.21.649858 (2025).
11. Vitale, R. et al. Protein language models are accidental taxonomists: emergent encoding of subcellular localization. *bioRxiv* https://doi.org/10.1101/2025.10.07.681002 (2025).
12. Gupta, A. et al. Topology-driven negative sampling enhances generalizability in protein–protein interaction prediction. *bioRxiv* https://doi.org/10.1101/2024.04.27.591478 (2024).
13. Zheng, X. et al. PRING: rethinking protein–protein interaction prediction from pairs to graphs. In *Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track* (2025).
14. Simon, E. & Zou, J. InterPLM: discovering interpretable features in protein language models via sparse autoencoders. *bioRxiv* https://doi.org/10.1101/2024.11.14.623630 (2024).
15. Adams, E. et al. From mechanistic interpretability to mechanistic biology: training, evaluating, and interpreting sparse autoencoders on protein language models. *bioRxiv* https://doi.org/10.1101/2025.02.06.636901 (2025).
16. Chen, T. & Guestrin, C. XGBoost: a scalable tree boosting system. In *Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining* 785–794 (2016).
17. Hollmann, N. et al. Accurate predictions on small data with a tabular foundation model. *Nature* **637**, 319–326 (2025).
18. Cornman, A., Tranzillo, M., Zulaybar, N. G., Bouzit, I. & Hwang, Y. Linear-time prediction of proteome-scale microbial protein interactions. *bioRxiv* https://doi.org/10.64898/2026.03.01.708874 (2026).
19. Hopf, T. A. et al. Sequence co-evolution gives 3D contacts and structures of protein complexes. *eLife* **3**, e03430 (2014).
20. Hayes, T. et al. Simulating 500 million years of evolution with a language model. *Science* **387**, 850–858 (2025).
21. ESM Team. ESM Cambrian: a parallel protein language model family for representation learning. *bioRxiv* https://doi.org/10.64898/2026.06.03.729735 (2026).
