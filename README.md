
# ParasGB

## Table of Contents

- [1. Dataset Details](#1-dataset-details)
- [2. Topology-to-Graph](#2-topology-to-graph)
- [3. Algorithms](#3-algorithms)
- [4. Limitations](#4-limitations)
- [5. Future Directions](#5-future-directions)
- [6. Experiment Details](#6-experiment-details)
- [7. Additional Task Results](#7-additional-task-results)
- [8. API Usage](#8-api-usage)

---

## 1. Dataset Details

### 1.1 SRAM

The SRAM subset uses **statistically aggregated features**, focusing on global topology and device distribution rather than preserving overly fine-grained local device details. This design controls feature dimensionality in ultra-large-graph scenarios while still retaining physically meaningful statistical information.

The main characteristics of the SRAM data are:

- extremely large graph size
- highly repetitive topology
- dense routing
- high coupling-capacitance density
- well suited for evaluating **scalability** and **computational efficiency**

The main subsets include:

- `ssram`: a baseline SRAM design used to verify whether a model can capture regular topology
- `digtime`: a digital timing-related logic module
- `timing ctrl`: an internal timing-control module in memory, with more complex topology
- `sandwich`: a stacked high-performance memory architecture with very high coupling density
- `ultra8t`: an 8T SRAM optimized for subthreshold low-power operation
- `array 128 32 8t`: one of the largest arrays in the dataset, used for extreme stress testing

### 1.3 Dataset Download

All raw graph data files are hosted on Hugging Face Datasets:

```text
https://huggingface.co/datasets/stomlvforever/ssram
```

After downloading, place the files under the following local paths:

```text
sram_rc/sram/
sram_rc/sram_r/
analog_rc/data/
```

#### SRAM Capacitance Dataset

| Dataset Name | Description | Download Link |
| --- | --- | --- |
| SSRAM | SRAM capacitance dataset for Cg and Cc tasks | [Download](https://huggingface.co/datasets/stomlvforever/ssram/resolve/main/sramc/ssram.pt) |
| DIGITAL_CLK_GEN | Digital timing capacitance dataset | [Download](https://huggingface.co/datasets/stomlvforever/ssram/resolve/main/sramc/digtime.pt) |
| TIMING_CTRL | Timing control capacitance dataset | [Download](https://huggingface.co/datasets/stomlvforever/ssram/resolve/main/sramc/timing_ctrl.pt) |
| ARRAY_128_32_8T | Large SRAM array capacitance dataset | [Download](https://huggingface.co/datasets/stomlvforever/ssram/resolve/main/sramc/array_128_32_8t.pt) |
| ULTRA8T | Ultra 8T SRAM capacitance dataset | [Download](https://huggingface.co/datasets/stomlvforever/ssram/resolve/main/sramc/ultra8t.pt) |
| SANDWICH_RAM | Sandwich SRAM capacitance dataset | [Download](https://huggingface.co/datasets/stomlvforever/ssram/resolve/main/sramc/sandwich.pt) |
| SP8192W | Specialized 8192 width capacitance dataset | [Download](https://huggingface.co/datasets/stomlvforever/ssram/resolve/main/sramc/sp8192w.pt) |

#### SRAM Resistance Dataset

| Dataset Name | Description | Download Link |
| --- | --- | --- |
| SSRAM_R | SRAM effective resistance dataset | [Download](https://huggingface.co/datasets/stomlvforever/ssram/resolve/main/sramr/ssram.pt) |
| DIGITAL_CLK_GEN_R | Digital timing effective resistance dataset | [Download](https://huggingface.co/datasets/stomlvforever/ssram/resolve/main/sramr/digtime.pt) |
| TIMING_CTRL_R | Timing control effective resistance dataset | [Download](https://huggingface.co/datasets/stomlvforever/ssram/resolve/main/sramr/timing_ctrl.pt) |
| ARRAY_128_32_8T_R | Large SRAM array effective resistance dataset | [Download](https://huggingface.co/datasets/stomlvforever/ssram/resolve/main/sramr/array_128_32_8t.pt) |
| ULTRA8T_R | Ultra 8T effective resistance dataset | [Download](https://huggingface.co/datasets/stomlvforever/ssram/resolve/main/sramr/ultra8t.pt) |
| SANDWICH_RAM_R | Sandwich SRAM effective resistance dataset | [Download](https://huggingface.co/datasets/stomlvforever/ssram/resolve/main/sramr/sandwich.pt) |

#### Analog RC Dataset

| Dataset Name | Description | Download Link |
| --- | --- | --- |
| Analog RC | Analog Cg and Reff dataset with 20 circuit cases | [Download](https://huggingface.co/datasets/stomlvforever/ssram/tree/main/analog) |

The node features of SRAM circuit graphs are defined as follows:

### Definition of SRAM Circuit Graph Node Features

#### Device

| Type | Feature | Definition | Index |
| --- | --- | --- | --- |
| Device | $M_{mos}$ | Multiplier of transistors | 0 |
| Device | $L$ | Length of the transistor | 1 |
| Device | $W$ | Width of the transistor | 2 |
| Device | $M_{res}$ | Multiplier of connected resistors | 3 |
| Device | $L_{res}$ | Length of resistor | 4 |
| Device | $W_{res}$ | Width of resistor | 5 |
| Device | $M_{cap}$ | Multiplier of connected capacitor | 6 |
| Device | $L_{r}$ | Length of capacitor | 7 |
| Device | $N_{r}$ | Number of capacitor fingers | 8 |
| Device | $N_{p}$ | Number of ports in the device instance | 9 |
| Device | $T$ | Type code of the device instance | 10 |

#### Net

| Type | Feature | Definition | Index |
| --- | --- | --- | --- |
| Net | $N_{mos}$ | Number of connected transistors | 0 |
| Net | $N_{g}$ | Number of connected gate terminals | 1 |
| Net | $N_{sd}$ | Number of connected source/drain terminals | 2 |
| Net | $N_{b}$ | Number of connected base terminals | 3 |
| Net | $W_{tot}$ | Total width of connected transistor | 4 |
| Net | $L_{tot}$ | Total length of connected transistor | 5 |
| Net | $N_{cap}$ | Number of connected capacitors | 6 |
| Net | $Lr_{tot}$ | Total length of connected capacitors | 7 |
| Net | $Nr_{tot}$ | Total number of connected capacitor fingers | 8 |
| Net | $N_{res}$ | Number of connected resistors | 9 |
| Net | $W_{tot,res}$ | Total width of connected resistors | 10 |
| Net | $L_{tot,res}$ | Total length of connected resistors | 11 |
| Net | $N_{port}$ | Number of connected ports | 12 |

#### Pin

| Type | Feature | Definition | Index |
| --- | --- | --- | --- |
| Pin | -- | Pin types (G/D/S/B for MOS) | 0 |

### 1.2 Analog

Unlike the SRAM dataset, the analog dataset contains smaller circuits but much more detailed device-level physical descriptions. Key parameters, such as device channel width `W`, length `L`, and the distance from the source/drain region to the isolation edge (the LDE effect), are included in the node-feature system. Circuit parasitic parameters are extracted using commercial PEX tools. Because analog circuits are highly sensitive to noise, even very small parasitic prediction errors can cause simulation results to deviate from expectations.

The dataset lists the sources and functional summaries of 20 analog cases, including:

- `ID 1`: an LVDS circuit that converts a low-frequency reference (5–27 MHz) into a high-frequency clock (100–700 MHz) with minimal phase noise for precise timing applications (Leung & Mok, 2003b).
- `ID 2`: an operational amplifier circuit that exploits quasi-linear temperature characteristics to generate a stable 0.4 V reference from an ultra-low 0.56 V supply while consuming only 4.8 μA current (Wang & Ye, 2006).
- `ID 3`: a bandgap reference circuit that uses a self-cascode composite transistor and a single resistor to generate a stable reference voltage close to the silicon bandgap, achieving a low temperature coefficient of 25.3 ppm/°C with only 25 μA current (Colombo et al., 2012).
- `ID 4`: a bandgap reference circuit that uses resistor subdivision and a resistorless method to generate a highly stable 910.88 mV reference voltage with an ultra-low temperature coefficient of 12.99 ppm/°C (Koh & Lee, 2014).
- `ID 5`: an LDO circuit that balances the temperature characteristics of N/P-type MOSFETs to generate a stable reference voltage for the LDO regulator, achieving a temperature coefficient of 36.9 ppm/°C with a low supply current of 9.7 μA (Leung & Mok, 2003a).
- `ID 6`: an LDO circuit that provides a stable output from a 1.8–4.5 V supply with fast transient response and minimal compensation capacitance (7 pF), supporting up to 100 mA load current with a 0.2 V dropout (Ho & Mok, 2010a).
- `ID 7`: an LDO circuit that balances the temperature characteristics of N/P-type MOSFETs to generate a stable reference voltage for the LDO regulator, achieving a temperature coefficient of 36.9 ppm/°C with a low supply current of 9.7 μA (Leung & Mok, 2003a).
- `ID 8`: an LDO circuit that uses an adaptive compensation buffer (ACB) to dynamically switch between pass transistors, enabling stable operation over a wide load range (0 to 30 mA) without external capacitors while maintaining a low quiescent current of 6 μA (Tan et al., 2025).
- `ID 9`: an LDO circuit that uses a three-loop architecture to achieve ultra-fast transient response (1.15 ns) while maintaining a clean power supply, with full-spectrum power-supply rejection (PSR > −12 dB up to 20 GHz) and only 50 μA quiescent current (Lu et al., 2015).
- `ID 10`: an operational amplifier circuit that provides a flexible, step-by-step method for balancing noise performance and power consumption, offering greater design control than previous methods and validated by multi-condition SPICE simulations (Mahattanakul & Chutichatuporn, 2005).
- `ID 11`: an LDO circuit that uses two parallel active-feedback paths to create two pole-zero pairs, providing superior stability and transient response compared with single-path methods, while supporting a 100 mA load with only 14 μA quiescent current (Li et al., 2020a).
- `ID 12`: an LDO circuit that uses a high-gain three-stage error amplifier to maintain accurate regulation even at ultra-low voltage (0.5 V supply) with an unsaturated pass transistor, achieving a current density of 11.4 A/mm² and a low-frequency PSR of −62 dB (Kim & Cho, 2023).
- `ID 13`: an operational amplifier circuit that replaces traditional Miller compensation with an active structure, eliminates the right-half-plane (RHP) zero, and introduces a left-half-plane (LHP) zero to cancel the first non-dominant pole, increasing the unity-gain frequency by 9.4× while significantly reducing compensation capacitance (Tan & Zhou, 2011).
- `ID 14`: a bandgap reference circuit that uses a combination of four MOSFETs, two lateral PNP transistors, and a well resistor to generate a stable 16 μA output current with a temperature coefficient of 105 ppm/°C, without requiring an external bandgap reference or trimming-process compensation (Osipov & Paul, 2017).
- `ID 15`: an LDO circuit that uses damping-zero compensation and a slew-rate enhancement circuit to achieve both stability and fast transients, with only 1.5 pF on-chip capacitance while supporting a 100 mA load and 200 mV dropout (Ho & Mok, 2010b).
- `ID 16`: an LDO circuit that uses a WCF circuit to maintain fast transient response and stable voltage regulation over a very wide range of load current (up to 100 mA) and load capacitance (470 pF to 10 nF), while consuming only 14.4 μA power (Wang et al., 2016).
- `ID 17`: an LDO circuit that uses a nested adaptive FVF structure to achieve ultra-fast transient response (handling load steps from 1 μA to 20 mA in only 10 ps), while significantly improving PSR (−58.52 dB at 1 MHz) and line regulation (Li et al., 2020b).
- `ID 18`: an LDO circuit that provides stable output with high DC gain (101 dB) and an accurate bandgap reference, supporting large-current loads (up to 450 mA) with only 0.5 V dropout while maintaining solid power-supply rejection (54.5 dB at 100 Hz) (Martínez-García et al., 2013).
- `ID 19`: a bandgap reference circuit that uses size-dependent effects to cancel process-induced threshold-voltage variation, achieving ultra-low power consumption of 192 pW and highly stable performance (0.53% process variation) without post-fabrication trimming (Ji et al., 2019).
- `ID 20`: a bandgap reference circuit that uses an ultra-low-power architecture to generate a stable reference voltage, with most of the 5 μA current dedicated to output, achieving a temperature coefficient below 10 ppm/°C from a 1 V supply without requiring a large-area operational amplifier (Edward, 2009).

### 1.4 Dataset Labels

Label distribution plots:

- **Analog Reff**: effective resistance distributions for 20 analog circuits

<table>
  <tr>
    <td align="center">
      <img src="IMGS/analog/edge_each_label/case1_RC_edge_normalized.png" width="180"><br>
      <sub>(a) Case 1</sub>
    </td>
    <td align="center">
      <img src="IMGS/analog/edge_each_label/case5_RC_edge_normalized.png" width="180"><br>
      <sub>(b) Case 2</sub>
    </td>
    <td align="center">
      <img src="IMGS/analog/edge_each_label/case7_RC_edge_normalized.png" width="180"><br>
      <sub>(c) Case 3</sub>
    </td>
    <td align="center">
      <img src="IMGS/analog/edge_each_label/case10_RC_edge_normalized.png" width="180"><br>
      <sub>(d) Case 4</sub>
    </td>
    <td align="center">
      <img src="IMGS/analog/edge_each_label/case11_RC_edge_normalized.png" width="180"><br>
      <sub>(e) Case 5</sub>
    </td>
  </tr>

  <tr>
    <td align="center">
      <img src="IMGS/analog/edge_each_label/case15_RC_edge_normalized.png" width="180"><br>
      <sub>(f) Case 6</sub>
    </td>
    <td align="center">
      <img src="IMGS/analog/edge_each_label/case17_RC_edge_normalized.png" width="180"><br>
      <sub>(g) Case 7</sub>
    </td>
    <td align="center">
      <img src="IMGS/analog/edge_each_label/case23_RC_edge_normalized.png" width="180"><br>
      <sub>(h) Case 8</sub>
    </td>
    <td align="center">
      <img src="IMGS/analog/edge_each_label/case29_RC_edge_normalized.png" width="180"><br>
      <sub>(i) Case 9</sub>
    </td>
    <td align="center">
      <img src="IMGS/analog/edge_each_label/case39_RC_edge_normalized.png" width="180"><br>
      <sub>(j) Case 10</sub>
    </td>
  </tr>

  <tr>
    <td align="center">
      <img src="IMGS/analog/edge_each_label/case42_RC_edge_normalized.png" width="180"><br>
      <sub>(k) Case 11</sub>
    </td>
    <td align="center">
      <img src="IMGS/analog/edge_each_label/case44_RC_edge_normalized.png" width="180"><br>
      <sub>(l) Case 12</sub>
    </td>
    <td align="center">
      <img src="IMGS/analog/edge_each_label/case45_RC_edge_normalized.png" width="180"><br>
      <sub>(m) Case 13</sub>
    </td>
    <td align="center">
      <img src="IMGS/analog/edge_each_label/case55_RC_edge_normalized.png" width="180"><br>
      <sub>(n) Case 14</sub>
    </td>
    <td align="center">
      <img src="IMGS/analog/edge_each_label/case58_RC_edge_normalized.png" width="180"><br>
      <sub>(o) Case 15</sub>
    </td>
  </tr>

  <tr>
    <td align="center">
      <img src="IMGS/analog/edge_each_label/case71_RC_edge_normalized.png" width="180"><br>
      <sub>(p) Case 16</sub>
    </td>
    <td align="center">
      <img src="IMGS/analog/edge_each_label/case72_RC_edge_normalized.png" width="180"><br>
      <sub>(q) Case 17</sub>
    </td>
    <td align="center">
      <img src="IMGS/analog/edge_each_label/case74_RC_edge_normalized.png" width="180"><br>
      <sub>(r) Case 18</sub>
    </td>
    <td align="center">
      <img src="IMGS/analog/edge_each_label/case75_RC_edge_normalized.png" width="180"><br>
      <sub>(s) Case 19</sub>
    </td>
    <td align="center">
      <img src="IMGS/analog/edge_each_label/case78_RC_edge_normalized.png" width="180"><br>
      <sub>(t) Case 20</sub>
    </td>
  </tr>
</table>

- **Analog Cg**: ground-capacitance distributions for 20 analog circuits

<h3 align="center">Analog RC Node Label Distributions</h3>

<table>
  <tr>
    <td align="center">
      <img src="IMGS/analog/node_each_label/case1_RC_normalized.png" width="180"><br>
      <sub>(a) Case 1</sub>
    </td>
    <td align="center">
      <img src="IMGS/analog/node_each_label/case5_RC_normalized.png" width="180"><br>
      <sub>(b) Case 2</sub>
    </td>
    <td align="center">
      <img src="IMGS/analog/node_each_label/case7_RC_normalized.png" width="180"><br>
      <sub>(c) Case 3</sub>
    </td>
    <td align="center">
      <img src="IMGS/analog/node_each_label/case10_RC_normalized.png" width="180"><br>
      <sub>(d) Case 4</sub>
    </td>
    <td align="center">
      <img src="IMGS/analog/node_each_label/case11_RC_normalized.png" width="180"><br>
      <sub>(e) Case 5</sub>
    </td>
  </tr>

  <tr>
    <td align="center">
      <img src="IMGS/analog/node_each_label/case15_RC_normalized.png" width="180"><br>
      <sub>(f) Case 6</sub>
    </td>
    <td align="center">
      <img src="IMGS/analog/node_each_label/case17_RC_normalized.png" width="180"><br>
      <sub>(g) Case 7</sub>
    </td>
    <td align="center">
      <img src="IMGS/analog/node_each_label/case23_RC_normalized.png" width="180"><br>
      <sub>(h) Case 8</sub>
    </td>
    <td align="center">
      <img src="IMGS/analog/node_each_label/case29_RC_normalized.png" width="180"><br>
      <sub>(i) Case 9</sub>
    </td>
    <td align="center">
      <img src="IMGS/analog/node_each_label/case39_RC_normalized.png" width="180"><br>
      <sub>(j) Case 10</sub>
    </td>
  </tr>

  <tr>
    <td align="center">
      <img src="IMGS/analog/node_each_label/case42_RC_normalized.png" width="180"><br>
      <sub>(k) Case 11</sub>
    </td>
    <td align="center">
      <img src="IMGS/analog/node_each_label/case44_RC_normalized.png" width="180"><br>
      <sub>(l) Case 12</sub>
    </td>
    <td align="center">
      <img src="IMGS/analog/node_each_label/case45_RC_normalized.png" width="180"><br>
      <sub>(m) Case 13</sub>
    </td>
    <td align="center">
      <img src="IMGS/analog/node_each_label/case55_RC_normalized.png" width="180"><br>
      <sub>(n) Case 14</sub>
    </td>
    <td align="center">
      <img src="IMGS/analog/node_each_label/case58_RC_normalized.png" width="180"><br>
      <sub>(o) Case 15</sub>
    </td>
  </tr>

  <tr>
    <td align="center">
      <img src="IMGS/analog/node_each_label/case71_RC_normalized.png" width="180"><br>
      <sub>(p) Case 16</sub>
    </td>
    <td align="center">
      <img src="IMGS/analog/node_each_label/case72_RC_normalized.png" width="180"><br>
      <sub>(q) Case 17</sub>
    </td>
    <td align="center">
      <img src="IMGS/analog/node_each_label/case74_RC_normalized.png" width="180"><br>
      <sub>(r) Case 18</sub>
    </td>
    <td align="center">
      <img src="IMGS/analog/node_each_label/case75_RC_normalized.png" width="180"><br>
      <sub>(s) Case 19</sub>
    </td>
    <td align="center">
      <img src="IMGS/analog/node_each_label/case78_RC_normalized.png" width="180"><br>
      <sub>(t) Case 20</sub>
    </td>
  </tr>
</table>

- **SRAM Cc**: coupling-capacitance distributions for 6 SRAM circuits

<table>
  <tr>
    <td align="center">
      <img src="IMGS/sram/edge_each_label/array_128_32_8t_normalized.png" width="160"><br>
      <sub>(a) Array_128_32_8t</sub>
    </td>
    <td align="center">
      <img src="IMGS/sram/edge_each_label/digtime_normalized.png" width="160"><br>
      <sub>(b) Digtime</sub>
    </td>
    <td align="center">
      <img src="IMGS/sram/edge_each_label/sandwich_normalized.png" width="160"><br>
      <sub>(c) Sandwich</sub>
    </td>
    <td align="center">
      <img src="IMGS/sram/edge_each_label/ssram_normalized.png" width="160"><br>
      <sub>(d) SSRAM</sub>
    </td>
    <td align="center">
      <img src="IMGS/sram/edge_each_label/timing_ctrl_normalized.png" width="160"><br>
      <sub>(e) Timing_Ctrl</sub>
    </td>
    <td align="center">
      <img src="IMGS/sram/edge_each_label/ultra8t_normalized.png" width="160"><br>
      <sub>(f) Ultra8t</sub>
    </td>
  </tr>
</table>

- **SRAM Reff**: effective-resistance distributions for 6 SRAM circuits

<table>
  <tr>
    <td align="center">
      <img src="IMGS/sram/edge_r_each_label/array_128_32_8t_normalized.png" width="160"><br>
      <sub>(a) Array_128_32_8t</sub>
    </td>
    <td align="center">
      <img src="IMGS/sram/edge_r_each_label/digtime_normalized.png" width="160"><br>
      <sub>(b) Digtime</sub>
    </td>
    <td align="center">
      <img src="IMGS/sram/edge_r_each_label/sandwich_normalized.png" width="160"><br>
      <sub>(c) Sandwich</sub>
    </td>
    <td align="center">
      <img src="IMGS/sram/edge_r_each_label/ssram_normalized.png" width="160"><br>
      <sub>(d) SSRAM</sub>
    </td>
    <td align="center">
      <img src="IMGS/sram/edge_r_each_label/timing_ctrl_normalized.png" width="160"><br>
      <sub>(e) Timing_Ctrl</sub>
    </td>
    <td align="center">
      <img src="IMGS/sram/edge_r_each_label/ultra8t_normalized.png" width="160"><br>
      <sub>(f) Ultra8t</sub>
    </td>
  </tr>
</table>

- **SRAM Cg**: ground-capacitance distributions for 6 SRAM circuits

<table>
  <tr>
    <td align="center">
      <img src="IMGS/sram/node_each_label/array_128_32_8t_normalized.png" width="160"><br>
      <sub>(a) Array_128_32_8t</sub>
    </td>
    <td align="center">
      <img src="IMGS/sram/node_each_label/digtime_normalized.png" width="160"><br>
      <sub>(b) Digtime</sub>
    </td>
    <td align="center">
      <img src="IMGS/sram/node_each_label/sandwich_normalized.png" width="160"><br>
      <sub>(c) Sandwich</sub>
    </td>
    <td align="center">
      <img src="IMGS/sram/node_each_label/ssram_normalized.png" width="160"><br>
      <sub>(d) SSRAM</sub>
    </td>
    <td align="center">
      <img src="IMGS/sram/node_each_label/timing_ctrl_normalized.png" width="160"><br>
      <sub>(e) Timing_Ctrl</sub>
    </td>
    <td align="center">
      <img src="IMGS/sram/node_each_label/ultra8t_normalized.png" width="160"><br>
      <sub>(f) Ultra8t</sub>
    </td>
  </tr>
</table>

Main observations:

- ground-capacitance labels generally show a **clear long-tailed distribution**
- SRAM labels span a wider range, making regression more difficult
- effective-resistance labels in analog circuits are sparser than capacitance labels and contain more outliers
- the number of coupling-capacitance edges in SRAM is much larger than the number of ground-capacitance node labels, making it a key scenario for evaluating non-local modeling ability

---

## 2. Topology-to-Graph

### 2.1 Analog Topology-to-Graph
![Analog2Graph](IMGS/analog2graph.png)

The conversion from analog circuit schematics to graph representations follows the framework shown in the figure. We model each circuit as a heterogeneous graph $\mathcal{G}=(\mathcal{V},\mathcal{E})$. The node set $\mathcal{V}$ contains three types of nodes: *device nodes* representing circuit components, *net nodes* representing interconnect wires, and *pin nodes* representing device terminals. The topological edges $\mathcal{E}_{\text{topo}}$ (shown as black lines) capture circuit connectivity derived from the schematic, specifically through *device-to-pin* and *pin-to-net* connections; these topological relations constitute the input structure obtained from the schematic-to-graph transformation.

In contrast, parasitic information is obtained from the extracted parasitic netlist. Blue *pin-to-pin* edges are treated as resistive edges, where the label corresponds to the effective resistance between two pins (see the algorithm in Section 6). In addition, we assign the total ground capacitance of each net as a node-level label on the corresponding net node. These parasitic labels serve as prediction targets in our benchmark.

### 2.2 Sram Topology-to-Graph
![Sram2Graph](IMGS/Sram2graph.png)

The figure illustrates the conversion process from an SRAM circuit schematic to its graph representation. Similar to Figure 1, the circuit is first modeled as a heterogeneous graph, $\mathcal{G}=(\mathcal{V},\mathcal{E})$, where the node set $\mathcal{V}$ consists of three types of nodes: **device nodes**, **net nodes**, and **pin nodes**.

The black topological edges represent the connectivity directly derived from the schematic, including **device-to-pin** and **pin-to-net** relations, which describe the fundamental topology of the SRAM cell.

On top of this topological structure, parasitic information extracted from the post-layout netlist is further incorporated into the graph. Specifically:

- **Blue pin-to-pin edges** denote resistive parasitics, where each edge label corresponds to the effective resistance between two pins.
- **Orange edges** denote coupling capacitance relations, which characterize the capacitive coupling effects between different pins or nets.
- In addition, each **net node** is associated with its **total ground capacitance** $C_g$ as a node-level label.

Therefore, the prediction targets in Figure 2 include not only resistance edge labels and net-level ground capacitance labels, but also coupling capacitance edge labels, enabling a more comprehensive representation of parasitic effects in SRAM circuits.

## 3. Algorithms

### 3.1 Matrix-Based Effective Resistance Calculation

> **Purpose.** Compute port-to-port effective resistance efficiently from the resistor netlist by constructing the nodal admittance matrix and querying pairwise resistances through its Cholesky factorization.

#### Overview

The procedure consists of three stages:

1. **Build the nodal admittance matrix**
   - Traverse all resistor elements in the net.
   - Convert each resistance value `r` into conductance `g = 1 / r`.
   - Update diagonal and off-diagonal entries according to Kirchhoff's Current Law (KCL).

2. **Construct the reduced invertible matrix**
   - Select one reference node, usually the ground node.
   - Remove the corresponding row and column from the admittance matrix.
   - Obtain the reduced admittance matrix `G_red`.

3. **Query effective resistances between ports**
   - Compute the Cholesky factor `L` such that `G_red = L L^T`.
   - Use `Z = L^{-1}` to derive pairwise effective resistances.
   - For each port pair, compute `R_eq = ||z_src - z_dst||^2`.

#### Pseudocode

<details>
<summary><strong>Algorithm 1. Matrix-Based Effective Resistance Calculation</strong></summary>

<br>

**Input:** Resistor list `R` of a net, port list `P`  
**Output:** Effective resistance list `L_out = {(src, dst, val)}`

```text
1:  L_out <- ∅
2:  V <- ExtractUniqueNodes(R)
3:  N <- |V|
4:  if N < 2 or |P| < 2 then
5:      return ∅
6:  end if
7:  M <- MapNodesToIndices(V)

8:  // Stage 1: Build admittance matrix
9:  G <- 0_{N×N}
10: for each (n1, n2, r) in R do
11:     g <- 1 / r
12:     u <- M[n1], v <- M[n2]
13:     G[u,u] <- G[u,u] + g
14:     G[v,v] <- G[v,v] + g
15:     G[u,v] <- G[u,v] - g
16:     G[v,u] <- G[v,u] - g
17: end for

18: ref <- N - 1
19: G_red <- G[0:ref, 0:ref]

20: // Stage 2: Cholesky factorization
21: Compute L such that G_red = L L^T
22: Z <- L^{-1}

23: // Stage 3: Port-to-port resistance extraction
24: for k <- 0 to |P| - 1 do
25:     for l <- k + 1 to |P| - 1 do
26:         src_id <- P[k]
27:         dst_id <- P[l]
28:         z_src <- column M[src_id] of Z
29:         z_dst <- column M[dst_id] of Z
30:         R_eq <- ||z_src - z_dst||^2
31:         append (src_id, dst_id, R_eq) to L_out
32:     end for
33: end for
34: return L_out
```

</details>

## 4. Limitations

Although ParasGB fills an important gap in benchmark research for circuit parasitic-effect modeling, several aspects still leave room for improvement as part of this early-stage exploration.

**Insufficient circuit-type coverage.** The current ParasGB dataset mainly covers SRAM and specific analog circuit modules. While these are representative, they do not cover all industrial design scenarios. For example, the layout styles and interconnect logic of complex digital circuits and very-large-scale SoC systems differ significantly from those of analog circuits. As a result, models trained on the current dataset may experience substantial performance degradation when directly transferred to digital-circuit scenarios.

**Challenges in accurate regression prediction.** Parasitic parameters exhibit pronounced long-tailed distributions, which makes extreme-value samples (very large or very small values) difficult to predict accurately. Although discretization (binning) reduces the difficulty of the task, it is essentially a compromise. Industrial applications still require high-precision numerical regression, and meeting that demand remains a core challenge for current algorithms.

**Lack of cross-technology-node validation.** The current dataset is mainly derived from specific advanced technologies. However, physical properties and design rules vary significantly across semiconductor technology generations, such as from 28 nm to 5 nm. Without large-scale cross-technology comparison data, it is difficult to fully validate model transferability to new technology nodes, which limits generalization across different foundries.

**Limited depth of physical-interaction modeling.** Current node features mainly include spatial coordinates and device-size information. In real chips, however, deeper physical effects such as local thermal behavior and complex electromagnetic coupling across multiple metal layers can also affect parasitic parameters. Although existing graph structures can model topological connectivity, they still provide insufficient depth for modeling these three-dimensional physical interactions, and some key physical features may therefore be overlooked.

## 5. Future Directions

### 5.1 Graph Foundation Model Pretraining

Conduct self-supervised pretraining on large-scale circuit graphs to learn general physical laws of circuits, and then adapt to downstream tasks with limited fine-tuning.

### 5.2 Enhanced Spatial-Geometric Awareness

Incorporate 3D layout information and metal-layer attributes more deeply into message passing so that the model can jointly understand:

- topological connectivity
- three-dimensional relative position
- routing-coupling relationships

### 5.3 Building a More Comprehensive Evaluation Platform

Future versions are planned to include:

- RF circuits
- high-speed interfaces
- large-scale digital logic modules
- layout data across multiple process nodes

### 5.4 Real-Time Guidance for Design Closure

Move from an offline benchmark toward online design assistance by providing parasitic warnings and optimization suggestions during the layout stage.

## 6. Experiment Details

### 6.1 Data Preprocessing

#### SRAM Task Filtering Rules

- `Cg` node task: keep `(1e-21, 1e-15) F`
- `Cc` edge task: keep `(1e-21, 1e-15) F`
- `Reff` edge task: filter zero values and apply global `1% - 99%` quantile normalization

#### Analog Task Filtering Rules

- `Cg` node task: keep `(0, 8e-13) F`
- `Reff` edge task: keep `[0, 700] Ω`

All classification tasks use **five equal-width bins**.

### 6.2 Baseline Details

This study selects a series of representative graph-learning models for comparative experiments to validate the challenge level of the ParasGB benchmark. The models are grouped as follows.

#### 1) Classical Message-Passing GNNs

These models are mainstream methods in graph learning. Their core idea is to learn local topological structure in circuits through feature propagation and aggregation among neighboring nodes. They offer efficient computation and relatively low memory consumption, making them some of the most widely used baselines in circuit-related tasks.

- `GCN` (Kipf & Welling, 2017): a classic graph convolutional network that aggregates neighbor features through Laplacian smoothing.
- `GAT` (Velicković et al., 2018): an attention-based graph neural network that adaptively weights neighboring nodes during message passing.
- `GraphSAGE` (Hamilton et al., 2017): an inductive graph representation model that learns node embeddings through neighborhood sampling and aggregation.
- `PNA` (Corso et al., 2020): a message-passing architecture that combines multiple aggregators and degree-scalers to improve expressive power.

#### 2) Graph Transformers

Unlike traditional models that only capture local neighborhoods, these models introduce global attention mechanisms and can model global topological dependencies across the entire circuit. For large circuit networks such as SRAM, which exhibit repetitive topology and long-range coupling effects, such models show stronger modeling potential.

- `SGFormer` (Wu et al., 2023): a lightweight graph transformer that efficiently models large-scale graphs through global attention.
- `PolyNormer` (Deng et al., 2024): a graph transformer with polynomial expressiveness and efficient long-range modeling capability.

#### 3) Circuit-Specific Models

These models are designed for task-specific pain points in the EDA domain, with targeted strategies for issues such as circuit-data scarcity and label imbalance.

- `CirGPS` (Shen et al., 2025c): a circuit-specific model that addresses circuit-data scarcity through subgraph sampling and few-shot/pretraining strategies.
- `CircuitGCL` (Shen et al., 2025a): a circuit-specific graph-contrastive framework designed to improve representation quality under challenging circuit-data distributions.

## 7. Additional Task Results

<h3 align="center">Performance of Different Models on Analog Circuits Ground Capacitance Node Classification Task</h3>

<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th colspan="2">1-4, 6, 8-12, 15-18</th>
      <th colspan="2">5</th>
      <th colspan="2">14</th>
      <th colspan="2">20</th>
    </tr>
    <tr>
      <th>Accuracy ↑</th>
      <th>F1-Score ↑</th>
      <th>Accuracy ↑</th>
      <th>F1-Score ↑</th>
      <th>Accuracy ↑</th>
      <th>F1-Score ↑</th>
      <th>Accuracy ↑</th>
      <th>F1-Score ↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>GCN</td>
      <td>0.8718</td>
      <td>0.5357</td>
      <td>0.9935</td>
      <td>0.4727</td>
      <td>0.9740</td>
      <td>0.3127</td>
      <td>0.9101</td>
      <td>0.3426</td>
    </tr>
    <tr>
      <td>GAT</td>
      <td>0.7949</td>
      <td>0.3000</td>
      <td>0.9941</td>
      <td>0.4991</td>
      <td>0.9761</td>
      <td>0.3239</td>
      <td>0.9101</td>
      <td>0.3426</td>
    </tr>
    <tr>
      <td>GraphSAGE</td>
      <td>0.7949</td>
      <td>0.3000</td>
      <td>0.9946</td>
      <td>0.8412</td>
      <td>0.9761</td>
      <td>0.3322</td>
      <td>0.9213</td>
      <td>0.5640</td>
    </tr>
    <tr>
      <td>PNA</td>
      <td>0.8974</td>
      <td>0.5804</td>
      <td>0.9941</td>
      <td>0.7028</td>
      <td>0.9783</td>
      <td>0.5189</td>
      <td>0.9326</td>
      <td>0.7926</td>
    </tr>
    <tr>
      <td>SGFormer</td>
      <td>0.8462</td>
      <td>0.4375</td>
      <td>0.9941</td>
      <td>0.4991</td>
      <td>0.9783</td>
      <td>0.4122</td>
      <td>0.9101</td>
      <td>0.3426</td>
    </tr>
    <tr>
      <td>PolyNormer</td>
      <td>0.8974</td>
      <td>0.6476</td>
      <td>0.9941</td>
      <td>0.7028</td>
      <td>0.9783</td>
      <td>0.4922</td>
      <td>0.9213</td>
      <td>0.5640</td>
    </tr>
    <tr>
      <td>CirGPS</td>
      <td>0.8460</td>
      <td>0.3806</td>
      <td>0.9911</td>
      <td>0.2492</td>
      <td>0.9740</td>
      <td>0.3215</td>
      <td>0.8876</td>
      <td>0.2714</td>
    </tr>
    <tr>
      <td>CircuitGCL</td>
      <td>0.8966</td>
      <td>0.9138</td>
      <td>0.9855</td>
      <td>0.3318</td>
      <td>0.9681</td>
      <td>0.3989</td>
      <td>0.8981</td>
      <td>0.3527</td>
    </tr>
  </tbody>
</table>

<h3 align="center">Performance of Different Models on SRAM Circuits Ground Capacitance Node Classification Task</h3>

<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th colspan="2">sram+digtime+timing_ctrl</th>
      <th colspan="2">sandwich</th>
      <th colspan="2">ultra8t</th>
      <th colspan="2">array_128_32_8t</th>
    </tr>
    <tr>
      <th>Accuracy ↑</th>
      <th>F1-Score ↑</th>
      <th>Accuracy ↑</th>
      <th>F1-Score ↑</th>
      <th>Accuracy ↑</th>
      <th>F1-Score ↑</th>
      <th>Accuracy ↑</th>
      <th>F1-Score ↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>GCN</td>
      <td>0.6385</td>
      <td>0.5542</td>
      <td>0.4000</td>
      <td>0.2437</td>
      <td>0.5410</td>
      <td>0.3321</td>
      <td>0.5045</td>
      <td>0.2688</td>
    </tr>
    <tr>
      <td>GAT</td>
      <td>0.6473</td>
      <td>0.2293</td>
      <td>0.3541</td>
      <td>0.2293</td>
      <td>0.4976</td>
      <td>0.2936</td>
      <td>0.4340</td>
      <td>0.3167</td>
    </tr>
    <tr>
      <td>GraphSAGE</td>
      <td>0.6480</td>
      <td>0.5799</td>
      <td>0.3343</td>
      <td>0.2105</td>
      <td>0.4932</td>
      <td>0.2853</td>
      <td>0.5909</td>
      <td>0.3575</td>
    </tr>
    <tr>
      <td>PNA</td>
      <td>0.6511</td>
      <td>0.5923</td>
      <td>0.4431</td>
      <td>0.2619</td>
      <td>0.5980</td>
      <td>0.3095</td>
      <td>0.4983</td>
      <td>0.3399</td>
    </tr>
    <tr>
      <td>SGFormer</td>
      <td>0.6193</td>
      <td>0.4345</td>
      <td>0.4594</td>
      <td>0.2734</td>
      <td>0.6159</td>
      <td>0.3373</td>
      <td>0.5200</td>
      <td>0.3310</td>
    </tr>
    <tr>
      <td>PolyNormer</td>
      <td>0.6511</td>
      <td>0.5895</td>
      <td>0.4398</td>
      <td>0.2532</td>
      <td>0.5740</td>
      <td>0.3218</td>
      <td>0.5654</td>
      <td>0.3503</td>
    </tr>
    <tr>
      <td>CirGPS</td>
      <td>0.9140</td>
      <td>0.2524</td>
      <td>0.8690</td>
      <td>0.2400</td>
      <td>0.9331</td>
      <td>0.2963</td>
      <td>0.9497</td>
      <td>0.3959</td>
    </tr>
    <tr>
      <td>CircuitGCL</td>
      <td>0.7258</td>
      <td>0.6344</td>
      <td>0.3867</td>
      <td>0.2751</td>
      <td>0.3517</td>
      <td>0.2724</td>
      <td>0.4043</td>
      <td>0.2853</td>
    </tr>
  </tbody>
</table>

<h3 align="center">Performance of Different Models on Analog Circuits Effective Resistance Edge Classification Task</h3>

<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th colspan="2">1-4, 6, 8-12, 15-18</th>
      <th colspan="2">5</th>
      <th colspan="2">14</th>
      <th colspan="2">20</th>
    </tr>
    <tr>
      <th>Accuracy ↑</th>
      <th>F1-Score ↑</th>
      <th>Accuracy ↑</th>
      <th>F1-Score ↑</th>
      <th>Accuracy ↑</th>
      <th>F1-Score ↑</th>
      <th>Accuracy ↑</th>
      <th>F1-Score ↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>GCN</td>
      <td>0.5314</td>
      <td>0.4462</td>
      <td>0.8474</td>
      <td>0.4678</td>
      <td>0.5007</td>
      <td>0.2875</td>
      <td>0.3316</td>
      <td>0.3323</td>
    </tr>
    <tr>
      <td>GAT</td>
      <td>0.5216</td>
      <td>0.4240</td>
      <td>0.8018</td>
      <td>0.3424</td>
      <td>0.2735</td>
      <td>0.1651</td>
      <td>0.1634</td>
      <td>0.1255</td>
    </tr>
    <tr>
      <td>GraphSAGE</td>
      <td>0.5164</td>
      <td>0.4238</td>
      <td>0.8415</td>
      <td>0.4505</td>
      <td>0.3934</td>
      <td>0.2437</td>
      <td>0.3381</td>
      <td>0.3395</td>
    </tr>
    <tr>
      <td>PNA</td>
      <td>0.4649</td>
      <td>0.2971</td>
      <td>0.8132</td>
      <td>0.3918</td>
      <td>0.3058</td>
      <td>0.1995</td>
      <td>0.1060</td>
      <td>0.0809</td>
    </tr>
    <tr>
      <td>SGFormer</td>
      <td>0.5199</td>
      <td>0.5163</td>
      <td>0.8684</td>
      <td>0.5575</td>
      <td>0.4024</td>
      <td>0.3316</td>
      <td>0.3638</td>
      <td>0.3618</td>
    </tr>
    <tr>
      <td>PolyNormer</td>
      <td>0.5017</td>
      <td>0.4229</td>
      <td>0.2636</td>
      <td>0.1747</td>
      <td>0.3565</td>
      <td>0.2049</td>
      <td>0.1859</td>
      <td>0.2265</td>
    </tr>
    <tr>
      <td>CirGPS</td>
      <td>0.6491</td>
      <td>0.6415</td>
      <td>0.6851</td>
      <td>0.3573</td>
      <td>0.2978</td>
      <td>0.2778</td>
      <td>0.3360</td>
      <td>0.2386</td>
    </tr>
    <tr>
      <td>CircuitGCL</td>
      <td>0.6383</td>
      <td>0.4678</td>
      <td>0.6440</td>
      <td>0.3078</td>
      <td>0.2881</td>
      <td>0.2349</td>
      <td>0.4204</td>
      <td>0.4288</td>
    </tr>
  </tbody>
</table>

<h3 align="center">Performance of Different Models on SRAM Circuits Ground Capacitance Node Regression Task</h3>

<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th colspan="2">sram+digtime+timing_ctrl</th>
      <th colspan="2">sandwich</th>
      <th colspan="2">ultra8t</th>
      <th colspan="2">array_128_32_8t</th>
    </tr>
    <tr>
      <th>MAE ↓</th>
      <th>R² ↑</th>
      <th>MAE ↓</th>
      <th>R² ↑</th>
      <th>MAE ↓</th>
      <th>R² ↑</th>
      <th>MAE ↓</th>
      <th>R² ↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>GCN</td>
      <td>0.0461</td>
      <td>0.8401</td>
      <td>0.1876</td>
      <td>0.1636</td>
      <td>0.0192</td>
      <td>0.5446</td>
      <td>0.1199</td>
      <td>-0.1967</td>
    </tr>
    <tr>
      <td>GAT</td>
      <td>0.0454</td>
      <td>0.8469</td>
      <td>0.1838</td>
      <td>0.2159</td>
      <td>0.0955</td>
      <td>0.5060</td>
      <td>0.1211</td>
      <td>-0.2518</td>
    </tr>
    <tr>
      <td>GraphSAGE</td>
      <td>0.0413</td>
      <td>0.8800</td>
      <td>0.1964</td>
      <td>0.0585</td>
      <td>0.1093</td>
      <td>0.2620</td>
      <td>0.2813</td>
      <td>-2.0157</td>
    </tr>
    <tr>
      <td>PNA</td>
      <td>0.0415</td>
      <td>0.8813</td>
      <td>0.1845</td>
      <td>0.3589</td>
      <td>0.0894</td>
      <td>0.4561</td>
      <td>0.2024</td>
      <td>-0.9577</td>
    </tr>
    <tr>
      <td>SGFormer</td>
      <td>0.0424</td>
      <td>0.8729</td>
      <td>0.2297</td>
      <td>-0.1733</td>
      <td>0.1619</td>
      <td>-0.3985</td>
      <td>0.2934</td>
      <td>-3.2101</td>
    </tr>
    <tr>
      <td>PolyNormer</td>
      <td>0.0423</td>
      <td>0.8667</td>
      <td>0.1938</td>
      <td>0.1186</td>
      <td>0.1563</td>
      <td>-0.5618</td>
      <td>0.1699</td>
      <td>-1.1344</td>
    </tr>
    <tr>
      <td>CirGPS</td>
      <td>0.0063</td>
      <td>0.9568</td>
      <td>0.0298</td>
      <td>0.6574</td>
      <td>0.0222</td>
      <td>0.7882</td>
      <td>0.0194</td>
      <td>0.8702</td>
    </tr>
    <tr>
      <td>CircuitGCL</td>
      <td>0.0511</td>
      <td>0.8792</td>
      <td>0.3558</td>
      <td>-0.4768</td>
      <td>0.3359</td>
      <td>-0.3439</td>
      <td>0.3314</td>
      <td>-0.6812</td>
    </tr>
  </tbody>
</table>

<h3 align="center">Performance of Different Models on SRAM Circuits Ground Capacitance Node Regression Task</h3>

<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th colspan="2">sram+digtime+timing_ctrl</th>
      <th colspan="2">sandwich</th>
      <th colspan="2">ultra8t</th>
      <th colspan="2">array_128_32_8t</th>
    </tr>
    <tr>
      <th>MAE ↓</th>
      <th>R² ↑</th>
      <th>MAE ↓</th>
      <th>R² ↑</th>
      <th>MAE ↓</th>
      <th>R² ↑</th>
      <th>MAE ↓</th>
      <th>R² ↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>GCN</td>
      <td>0.0461</td>
      <td>0.8401</td>
      <td>0.1876</td>
      <td>0.1636</td>
      <td>0.0192</td>
      <td>0.5446</td>
      <td>0.1199</td>
      <td>-0.1967</td>
    </tr>
    <tr>
      <td>GAT</td>
      <td>0.0454</td>
      <td>0.8469</td>
      <td>0.1838</td>
      <td>0.2159</td>
      <td>0.0955</td>
      <td>0.5060</td>
      <td>0.1211</td>
      <td>-0.2518</td>
    </tr>
    <tr>
      <td>GraphSAGE</td>
      <td>0.0413</td>
      <td>0.8800</td>
      <td>0.1964</td>
      <td>0.0585</td>
      <td>0.1093</td>
      <td>0.2620</td>
      <td>0.2813</td>
      <td>-2.0157</td>
    </tr>
    <tr>
      <td>PNA</td>
      <td>0.0415</td>
      <td>0.8813</td>
      <td>0.1845</td>
      <td>0.3589</td>
      <td>0.0894</td>
      <td>0.4561</td>
      <td>0.2024</td>
      <td>-0.9577</td>
    </tr>
    <tr>
      <td>SGFormer</td>
      <td>0.0424</td>
      <td>0.8729</td>
      <td>0.2297</td>
      <td>-0.1733</td>
      <td>0.1619</td>
      <td>-0.3985</td>
      <td>0.2934</td>
      <td>-3.2101</td>
    </tr>
    <tr>
      <td>PolyNormer</td>
      <td>0.0423</td>
      <td>0.8667</td>
      <td>0.1938</td>
      <td>0.1186</td>
      <td>0.1563</td>
      <td>-0.5618</td>
      <td>0.1699</td>
      <td>-1.1344</td>
    </tr>
    <tr>
      <td>CirGPS</td>
      <td>0.0063</td>
      <td>0.9568</td>
      <td>0.0298</td>
      <td>0.6574</td>
      <td>0.0222</td>
      <td>0.7882</td>
      <td>0.0194</td>
      <td>0.8702</td>
    </tr>
    <tr>
      <td>CircuitGCL</td>
      <td>0.0511</td>
      <td>0.8792</td>
      <td>0.3558</td>
      <td>-0.4768</td>
      <td>0.3359</td>
      <td>-0.3439</td>
      <td>0.3314</td>
      <td>-0.6812</td>
    </tr>
  </tbody>
</table>

## 8. API Usage

### 8.1 Standardized Evaluation Protocol

ParasGB is deeply integrated with **PyTorch Geometric (PyG)** so that researchers can complete the following steps with only a small amount of code:

- dataset download
- graph data preprocessing
- task-level data loading
- standardized evaluation

To support ultra-large circuit graphs, the toolkit provides:

- `NeighborLoader`: for node-level tasks
- `LinkNeighborLoader`: for edge-level tasks

Both loaders support caching preprocessed graph data on first use, which reduces repeated computation, improves reproducibility, and avoids experimental discrepancies caused by differences in preprocessing logic.

In addition, ParasGB provides a unified `Evaluator` module that follows the OGB style and automatically reports standardized metrics:

- classification tasks: Accuracy / F1
- regression tasks: MAE / R²

The dataset object also exposes metadata such as:

- train/test splits
- number of nodes and edges
- label distributions

This makes it easier to compare the real performance of different models across different tasks in a transparent way.

### 8.2 ParasGB Usage

The core goal of ParasGB is to **lower the barrier to parasitic-parameter learning research**. Its usage style is intentionally close to PyG. Researchers only need to specify:

- dataset name
- task level (`node` / `edge`)
- task type (`classification` / `regression`)

The system will then automatically complete raw file download and feature preprocessing.

For SRAM graphs with tens of millions of nodes, the toolkit provides subgraph sampling for limited-memory settings to ensure both training efficiency and stability.

### 8.3 Task Name Reference
#### SRAM

- `cg_regr`: node-level ground capacitance regression
- `cg_class`: node-level ground capacitance classification
- `cc_regr`: edge-level coupling capacitance regression
- `cc_class`: edge-level coupling capacitance classification
- `r_regr`: edge-level effective resistance regression
- `r_class`: edge-level effective resistance classification

#### Analog

- `cg_regr`: node-level ground capacitance regression
- `cg_class`: node-level ground capacitance classification
- `r_regr`: edge-level effective resistance regression
- `r_class`: edge-level effective resistance classification

### 8.4 Task Launch Commands

#### SRAM

Use `sram_rc/rcg/main.py` for the SRAM tasks. The default split trains on `ssram+digtime+timing_ctrl` and tests on `sandwich+ultra8t+array_128_32_8t`.

```bash
cd sram_rc/rcg

# Cg node regression
python main.py --data_type c --task_level node --task regression --model gcn --gpu 0

# Cg node classification
python main.py --data_type c --task_level node --task classification --num_classes 5 --model gcn --gpu 0

# Cc edge regression
python main.py --data_type c --task_level edge --task regression --model gcn --gpu 0

# Cc edge classification
python main.py --data_type c --task_level edge --task classification --num_classes 5 --model gcn --gpu 0

# Reff edge regression
python main.py --data_type r --task_level edge --task regression --model gcn --gpu 0

# Reff edge classification
python main.py --data_type r --task_level edge --task classification --num_classes 2 --model gcn --gpu 0
```

CircuitGCL configuration:

```bash
cd sram_rc/CircuitGCL

# Cg node regression
python main.py --data_type c --task_level node --task regression --model clustergcn --num_gnn_layers 4 --num_head_layers 3 --hid_dim 96 --dropout 0.4 --lr 0.00005 --batch_size 128 --gpu 0

# Cg node classification
python main.py --data_type c --task_level node --task classification --num_classes 5 --model clustergcn --num_gnn_layers 4 --num_head_layers 3 --hid_dim 96 --dropout 0.4 --lr 0.00005 --batch_size 128 --gpu 0

# Cc edge regression
python main.py --data_type c --task_level edge --task regression --model clustergcn --num_gnn_layers 4 --num_head_layers 3 --hid_dim 96 --dropout 0.4 --lr 0.00005 --batch_size 128 --gpu 0

# Cc edge classification
python main.py --data_type c --task_level edge --task classification --num_classes 5 --model clustergcn --num_gnn_layers 4 --num_head_layers 3 --hid_dim 96 --dropout 0.4 --lr 0.00005 --batch_size 128 --gpu 0

# Reff edge regression
python main.py --data_type r --task_level edge --task regression --model clustergcn --num_gnn_layers 4 --num_head_layers 3 --hid_dim 96 --dropout 0.4 --lr 0.00005 --batch_size 128 --gpu 0

# Reff edge classification
python main.py --data_type r --task_level edge --task classification --num_classes 2 --model clustergcn --num_gnn_layers 4 --num_head_layers 3 --hid_dim 96 --dropout 0.4 --lr 0.00005 --batch_size 128 --gpu 0
```

CirGPS configuration:

```bash
cd sram_rc/Cirgps

# Cg node regression
python main.py --data_type c --task_level node --task regression --model clustergcn --num_gnn_layers 4 --num_head_layers 2 --hid_dim 144 --dropout 0.3 --lr 0.0001 --batch_size 128 --gpu 0

# Cg node classification
python main.py --data_type c --task_level node --task classification --num_classes 5 --model clustergcn --num_gnn_layers 4 --num_head_layers 2 --hid_dim 144 --dropout 0.3 --lr 0.0001 --batch_size 128 --gpu 0

# Cc edge regression
python main.py --data_type c --task_level edge --task regression --model clustergcn --num_gnn_layers 4 --num_head_layers 2 --hid_dim 144 --dropout 0.3 --lr 0.0001 --batch_size 128 --gpu 0

# Cc edge classification
python main.py --data_type c --task_level edge --task classification --num_classes 5 --model clustergcn --num_gnn_layers 4 --num_head_layers 2 --hid_dim 144 --dropout 0.3 --lr 0.0001 --batch_size 128 --gpu 0

# Reff edge regression
python main.py --data_type r --task_level edge --task regression --model clustergcn --num_gnn_layers 4 --num_head_layers 2 --hid_dim 144 --dropout 0.3 --lr 0.0001 --batch_size 128 --gpu 0

# Reff edge classification
python main.py --data_type r --task_level edge --task classification --num_classes 2 --class_boundaries 0.5 --model clustergcn --num_gnn_layers 4 --num_head_layers 2 --hid_dim 144 --dropout 0.3 --lr 0.0001 --batch_size 128 --gpu 0
```

#### Analog

Use `analog_rc/change/main.py` for the analog tasks. The default split trains on `1+2+3+6+8+9+10+11+12+15+16+17+18` and tests on `5+14+20`.

```bash
cd analog_rc/change

# Cg node regression
python main.py --task_level node --task regression --model sage --gpu 0

# Cg node classification
python main.py --task_level node --task classification --num_classes 5 --model sage --gpu 0

# Reff edge regression
python main.py --task_level edge --task regression --model sage --edge_sample_rate 0.6 --gpu 0

# Reff edge classification
python main.py --task_level edge --task classification --num_classes 5 --model sage --edge_sample_rate 0.6 --gpu 0
```

CircuitGCL configuration:

```bash
cd analog_rc/CircuitGCL

# Cg node regression
python main.py --task_level node --task regression --train_dataset "1+2+3+6+8+9+10+11+12+15+16+17+18" --test_dataset "5+14+20" --model sage --num_gnn_layers 4 --num_head_layers 2 --hid_dim 144 --dropout 0.4 --lr 0.0001 --epochs 200 --act_fn leakyrelu --gpu 0

# Cg node classification
python main.py --task_level node --task classification --train_dataset "1+2+3+6+8+9+10+11+12+15+16+17+18" --test_dataset "5+14+20" --num_classes 5 --model sage --num_gnn_layers 4 --num_head_layers 2 --hid_dim 144 --dropout 0.4 --lr 0.0001 --epochs 200 --act_fn leakyrelu --gpu 0

# Reff edge regression
python main.py --task_level edge --task regression --train_dataset "1+2+3+6+8+9+10+11+12+15+16+17+18" --test_dataset "5+14+20" --model sage --num_gnn_layers 4 --num_head_layers 2 --hid_dim 144 --dropout 0.4 --lr 0.0001 --epochs 200 --act_fn leakyrelu --gpu 0

# Reff edge classification
python main.py --task_level edge --task classification --train_dataset "1+2+3+6+8+9+10+11+12+15+16+17+18" --test_dataset "5+14+20" --num_classes 5 --model sage --num_gnn_layers 4 --num_head_layers 2 --hid_dim 144 --dropout 0.4 --lr 0.0001 --epochs 200 --act_fn leakyrelu --gpu 0
```

CirGPS configuration:

```bash
cd analog_rc/Cirgps

# Cg node regression
python main.py --task_level node --task regression --train_dataset "1+2+3+6+8+9+10+11+12+15+16+17+18" --test_dataset "5+14+20" --model clustergcn --num_gnn_layers 4 --num_head_layers 2 --hid_dim 84 --dropout 0.4 --lr 0.0001 --num_hops 3 --act_fn leakyrelu --gpu 0

# Cg node classification
python main.py --task_level node --task classification --train_dataset "1+2+3+6+8+9+10+11+12+15+16+17+18" --test_dataset "5+14+20" --num_classes 5 --model clustergcn --num_gnn_layers 4 --num_head_layers 2 --hid_dim 84 --dropout 0.4 --lr 0.0001 --num_hops 3 --act_fn leakyrelu --gpu 0

# Reff edge regression
python main.py --task_level edge --task regression --train_dataset "1+2+3+6+8+9+10+11+12+15+16+17+18" --test_dataset "5+14+20" --model clustergcn --num_gnn_layers 4 --num_head_layers 2 --hid_dim 84 --dropout 0.4 --lr 0.0001 --num_hops 3 --act_fn leakyrelu --gpu 0

# Reff edge classification
python main.py --task_level edge --task classification --train_dataset "1+2+3+6+8+9+10+11+12+15+16+17+18" --test_dataset "5+14+20" --num_classes 5 --model clustergcn --num_gnn_layers 4 --num_head_layers 2 --hid_dim 84 --dropout 0.4 --lr 0.0001 --num_hops 3 --act_fn leakyrelu --gpu 0
```

---
