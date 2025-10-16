# ThinkAct: Vision-Language-Action Reasoning via Reinforced Visual Latent Planning

### Abstract
Vision-language-action (VLA) reasoning tasks require agents to interpret multimodal instructions, perform long-horizon planning, and act adaptively in dynamic environments.  
ビジョン・言語・行動（VLA）推論タスクでは、エージェントがマルチモーダルな指示を解釈し、長期的な計画を行い、動的な環境に適応して行動することが求められます。

Existing approaches typically train VLA models in an end-to-end fashion, directly mapping inputs to actions without explicit reasoning, which hinders their ability to plan over multiple steps or adapt to complex task variations.  
既存のアプローチは通常、VLAモデルをエンドツーエンド方式でトレーニングし、入力を明示的な推論なしに直接アクションにマッピングするため、複数ステップにわたる計画や複雑なタスクのバリエーションへの適応能力が妨げられます。

In this paper, we propose ThinkAct, a dual-system framework that bridges high-level reasoning with low-level action execution via reinforced visual latent planning.  
本稿では、強化学習を用いた視覚的潜在プランニングを通じて、高レベルの推論と低レベルの行動実行を橋渡しするデュアルシステムフレームワーク「ThinkAct」を提案します。

ThinkAct trains a multimodal LLM to generate embodied reasoning plans guided by reinforcing action-aligned visual rewards based on goal completion and trajectory consistency.  
ThinkActは、目標の達成と軌道の一貫性に基づいた行動に連携する視覚的報酬を強化学習の指針とすることで、具体化された推論計画を生成するようマルチモーダルLLMをトレーニングします。

These reasoning plans are compressed into a visual plan latent that conditions a downstream action model for robust action execution on target environments.  
これらの推論計画は、ターゲット環境での頑健な行動実行のために下流のアクションモデルを条件付ける視覚的計画の潜在表現に圧縮されます。

Extensive experiments on embodied reasoning and robot manipulation benchmarks demonstrate that ThinkAct enables few-shot adaptation, long-horizon planning, and self-correction behaviors in complex embodied AI tasks.  
具体化された推論とロボット操作のベンチマークに関する広範な実験により、ThinkActが複雑な具体化AIタスクにおいて、 few-shot 適応、長期計画、および自己修正行動を可能にすることが実証されています。

### **1\. Introduction**

Recent advances in multimodal large language models (MLLMs) have led to impressive progress on various tasks requiring the understanding of multimodal inputs, such as visual question answering and image/video captioning.  
マルチモーダル大規模言語モデル（MLLM）の最近の進歩は、視覚的質問応答や画像/ビデオキャプションなど、マルチモーダル入力の理解を必要とするさまざまなタスクで目覚ましい進歩をもたらしました。

However, while multimodal content can now be effectively perceived and interpreted, conducting multi-step planning for long-horizon user goals and then interacting with dynamic environments remains challenging for frontier MLLMs.  
しかし、マルチモーダルコンテンツを効果的に認識し解釈できるようになった一方で、長期的なユーザー目標のための複数ステップの計画を立て、その後動的な環境と対話することは、最先端のMLLMにとっても依然として困難です。

Therefore, enabling the vision-language foundation models with action awareness and embodied reasoning capabilities unleashes a wide range of physical Al applications (e.g., robotics and AR assistance), and draws significant attention from both academics and industry.  
したがって、視覚言語基盤モデルに行動認識と具体化された推論能力を持たせることは、幅広い物理的なAIアプリケーション（ロボット工学やAR支援など）を解き放ち、学界と産業界の両方から大きな注目を集めています。

To bridge action with vision-language modalities, several works learn vision-language-action (VLA) models by initializing from pre-trained MLLMs and training on large-scale robotic demonstrations.  
行動と視覚言語モダリティを橋渡しするために、いくつかの研究では、事前に訓練されたMLLMから初期化し、大規模なロボットデモンストレーションで訓練することによって、視覚言語行動（VLA）モデルを学習しています。

For example, OpenVLA builds upon MLLMs with post-training on large-scale robot demonstrations, while TraceVLA further applies visual traces prompting to enhance spatial context understanding.  
例えば、OpenVLAは大規模なロボットデモンストレーションに関する事後訓練を伴うMLLMを基盤としており、一方TraceVLAは空間的文脈理解を強化するために視覚的トレースプロンプティングをさらに適用しています。

Despite promising on short-horizon skills, the crucial capabilities to reason in diverse visual scenes and enable long-horizon planning remain limited due to the end-to-end fashion from visual and textual inputs to low-level actions.  
短期的なスキルについては有望であるにもかかわらず、多様な視覚シーンで推論し、長期的な計画を可能にするという重要な能力は、視覚的およびテキスト入力から低レベルのアクションへのエンドツーエンド方式のために制限されたままです。

To equip VLAs with the ability to solve complex embodied tasks, recent works have explored incorporating explicit chain-of-thought (CoT) prompting as an intermediate step-by-step guidance.  
VLAに複雑な身体化タスクを解決する能力を持たせるため、最近の研究では、中間的なステップバイステップのガイダンスとして、明示的な思考の連鎖（CoT）プロンプティングを組み込むことが検討されています。

For instance, ECOT and RAD introduce data curation pipelines to generate intermediate steps and decomposed plans by prompting off-the-shelf MLLMs.  
例えば、ECOTとRADは、既製のMLLMをプロンプティングすることによって、中間ステップと分解された計画を生成するためのデータキュレーションパイプラインを導入しています。

Once the annotated CoT traces are obtained, VLAs are trained to predict intermediate steps via fully supervised fine-tuning (SFT).  
注釈付きCoTトレースが取得されると、VLAは完全に監視されたファインチューニング（SFT）を介して中間ステップを予測するようにトレーニングされます。

However, due to the high cost of producing high-quality reasoning traces, the resulting models are prone to overfitting to a specific visual scenes or reasoning patterns.  
しかし、高品質な推論トレースを作成するにはコストがかかるため、結果として得られるモデルは、特定の視覚シーンや推論パターンに過剰適合する傾向があります。

Recently, reinforcement learning (RL) has demonstrated significant potential to incentivize reasoning behaviors in LLMs by exploring the thinking trace that maximizes reward signals instead of solely relying on fully supervised CoT annotations.  
最近、強化学習（RL）は、完全に教師ありのCoTアノテーションにのみ依存するのではなく、報酬信号を最大化する思考トレースを探索することによって、LLMの推論行動を動機付ける大きな可能性を示しています。

Inspired by this paradigm, several vision-language models have applied RL-based reasoning to multimodal tasks.  
このパラダイムに触発され、いくつかの視覚言語モデルは、RLベースの推論をマルチモーダルタスクに適用しています。

For example, Video-R1 adopts R1-style RL optimization to induce the CoT traces by verifiable answer accuracy with format correctness.  
たとえば、Video-R1はR1スタイルのRL最適化を採用して、フォーマットの正確さを備えた検証可能な回答精度によってCoTトレースを誘導します。

While this manner enables long-form reasoning without step-level supervision, the reliance on QA-style reward signals limits their ability to support long-horizon planning and makes it difficult to connect reasoning with real-world action execution.  
この方法はステップレベルの監視なしで長文の推論を可能にしますが、QA形式の報酬信号に依存するため、長期計画をサポートする能力が制限され、推論と現実世界の行動実行を結びつけることが困難になります。

In this paper, we propose ThinkAct, which aims to enable MLLMs with the capability to reason before acting in physical environments.  
本稿では、物理環境で行動する前に推論する能力をMLLMに持たせることを目的としたThinkActを提案します。

To address vision-language-action reasoning tasks, ThinkAct adopts a dual-system architecture that connects structured reasoning with executable actions.  
視覚言語行動の推論タスクに取り組むため、ThinkActは構造化された推論と実行可能な行動を結びつけるデュアルシステムアーキテクチャを採用しています。

Specifically, we incentivize MLLMs to perform long-horizon planning by advancing reinforcement learning with an action-aligned reward, derived from visual goal completion and trajectory distribution matching.  
具体的には、視覚的な目標達成と軌道分布マッチングから導き出される行動に合わせた報酬を用いた強化学習を進めることで、MLLMに長期的な計画を実行させるように動機付けます。

Our ThinkAct leverages human and robot videos to elicit embodied reasoning that is grounded in visual observations.  
私たちのThinkActは、人間とロボットのビデオを活用して、視覚的観察に基づいた身体化された推論を引き出します。

To bridge reasoning and execution, we compress intermediate reasoning steps into a compact latent trajectory that captures high-level intent and allows efficient adaptation of the downstream action network to new environments.  
推論と実行を橋渡しするために、中間的な推論ステップを、高レベルの意図を捉え、下流のアクションネットワークの新しい環境への効率的な適応を可能にするコンパクトな潜在軌道に圧縮します。

By reinforcing structured reasoning and grounding it in real-world actions, ThinkAct tackles long-horizon manipulation tasks while unleashing few-shot action adaptation and self-correction behavior in physical Al scenarios, as shown in Fig.1\.  
構造化された推論を強化し、それを現実世界の行動に接地させることで、ThinkActは長期的な操作タスクに取り組みながら、図1に示すように、physical AI シナリオにおける few-shot の行動適応と自己修正行動を解き放ちます。

Our main contributions are summarized as follows:  
私たちの主な貢献は以下の通りです。

 - We propose ThinkAct, a dual-system framework that mutually enhances action execution and visual-grounded embodied reasoning connected by visual latent planning.  <br>私たちは、視覚的潜在プランニングによって接続された、行動実行と視覚接地された身体化推論を相互に強化するデュアルシステムフレームワークであるThinkActを提案します。

 - We leverage the visual feedback of goal completion and trajectory alignment as action-aligned rewards to allow long-horizon reasoning grounded in the embodied scene.  <br>具体化されたシーンに接地された長期的な推論を可能にするために、目標達成と軌道アライメントの視覚的フィードバックを行動に連携した報酬として活用します。

 - We advance visual latent planning to steer downstream action execution by providing reasoning-enhanced trajectory guidance across diverse environments.  <br>多様な環境にわたって推論強化された軌道ガイダンスを提供することにより、下流の行動実行を操縦するための視覚的潜在計画を進めます。

 - We demonstrate that our learned reasoning VLA enables capabilities of few-shot adaptation, long-horizon planning, and self-correction across diverse embodied manipulation tasks.  <br>私たちが学習した推論VLAが、多様な身体化操作タスクにおいて、few-shot適応、長期計画、自己修正の能力を可能にすることを実証します。


### **2\. Related Works**

#### **2.1. Vision-Language-Action Models**

Recent efforts have adapted vision-language models (VLMs) for action-centric tasks by post-training on curated instruction-following data.  
最近の取り組みでは、キュレーションされた指示追従データに関する事後トレーニングによって、行動中心のタスクに視覚言語モデル（VLM）を適応させています。

For example, RoboPoint and LLARVA leverage point and visual trajectory into textual prompts to augment LLMs with spatial-action understanding ability.  
例えば、RoboPointとLLARVAは、点と視覚的な軌跡をテキストプロンプトに活用して、LLMに空間的行動理解能力を付加します。

AHA enhances failure detection ability in robotic manipulation by formulating it as a free-form question-answering task, training on synthetic failure data generated by perturbing successful trajectories.  
AHAは、成功した軌道を摂動させることによって生成された合成失敗データで訓練し、自由形式の質疑応答タスクとして定式化することによって、ロボット操作における失敗検出能力を強化します。

Although effective in a specific domains, these approaches depend on sophisticatedly curated data and struggle to generalize beyond their training distributions.  
特定のドメインでは効果的ですが、これらのアプローチは洗練されたキュレーションデータに依存しており、トレーニング分布を超えて一般化することに苦労しています。

To improve scalability, recent vision-language-action (VLA) models adopt large-scale robot datasets to train models directly on diverse demonstrations.  
スケーラビリティを向上させるために、最近の視覚言語行動（VLA）モデルは、大規模なロボットデータセットを採用して、多様なデモンストレーションでモデルを直接トレーニングします。

OpenVLA learns from pre-trained VLMs with robot trajectories for generalist action execution, while TraceVLA and HAMSTER enhance spatial-action awareness by incorporating visual traces.  
OpenVLAは、汎用的な行動実行のためにロボットの軌跡を持つ事前訓練されたVLMから学習し、TraceVLAとHAMSTERは視覚的な軌跡を組み込むことで空間行動認識を強化します。

However, these models predict actions directly from vision and language inputs, often bypassing structured planning or intermediate reasoning.  
しかし、これらのモデルは、視覚と言語の入力から直接行動を予測するため、構造化された計画や中間的な推論をしばしば迂回します。

As a result, their capability to handle complex instructions, long-horizon goals, or out-of-distribution scenarios remains limited.  
その結果、複雑な指示、長期的な目標、または分布外のシナリオを処理する能力は限られたままです。

#### **2.2. Reasoning in Vision-Language-(Action) Models**

Chain-of-thought (CoT) prompting has significantly improved the multi-step reasoning ability of LLMs across math, coding, and question-answering tasks.  
思考の連鎖（CoT）プロンプティングは、数学、コーディング、質疑応答の各タスクにわたるLLMの複数ステップの推論能力を大幅に向上させました。

Motivated by these advances, recent works extend reasoning capabilities to vision-language-action (VLA) models for embodied tasks.  
これらの進歩に動機付けられ、最近の研究では、身体化されたタスクのために、推論能力を視覚言語行動（VLA）モデルに拡張しています。

ECoT synthesizes intermediate subgoals via prompting and applies supervised fine-tuning to teach VLAs to reason before acting.  
ECoTは、プロンプティングを介して中間的なサブゴールを合成し、VLAが行動する前に推論するように教えるために、教師ありファインチューニングを適用します。

RAD leverages action-free human videos to curate reasoning traces by prompting off-the-shelf LLMs and learn to map reasoning to real actions using robot data.  
RADは、既製のLLMをプロンプティングすることで推論トレースをキュレートするために、アクションフリーのヒューマンビデオを活用し、ロボットデータを使用して推論を実際のアクションにマッピングすることを学習します。

On the other hand, CoT-VLA replaces linguistic CoT with visual subgoal frames generated ahead of action prediction.  
一方、CoT-VLAは、言語的なCoTを、行動予測の前に生成される視覚的なサブゴールフレームに置き換えます。

However, they depend on either curated CoT supervision or task-specific video generation, limiting their scalability.  
しかし、それらはキュレーションされたCoT監督またはタスク固有のビデオ生成のいずれかに依存しており、スケーラビリティが制限されています。

Inspired by the recent success of RL-optimized reasoning models, several approaches adopt GRPO optimization to guide CoT generation in vision-language tasks using verifiable rewards.  
RLに最適化された推論モデルの最近の成功に触発され、いくつかのアプローチでは、検証可能な報酬を使用して視覚言語タスクにおけるCoT生成をガイドするためにGRPO最適化を採用しています。

However, their QA-formatted rewards cannot fully support long-horizon planning or establish grounding between reasoning and action execution.  
しかし、彼らのQA形式の報酬は、長期的な計画を完全にサポートしたり、推論と行動実行の間の根拠を確立したりすることはできません。

To unify structured CoT reasoning with embodied decision-making, we introduce ThinkAct, which leverages action-aligned reinforcement learning and visual latent planning to connect embodied reasoning with real-world action in VLA tasks.  
構造化されたCoT推論と身体化された意思決定を統合するために、我々は、VLAタスクにおける身体化された推論を現実世界の行動と結びつけるために、行動に整合した強化学習と視覚的潜在計画を活用するThinkActを導入します。

### **3\. Method**

#### **3.1. Problem Formulation**

We first define the setting and notations for vision-language-action (VLA) reasoning tasks.  
まず、視覚言語行動（VLA）推論タスクの設定と表記法を定義します。

At each timestep t, the model receives a visual observation $o_{t}$ and a textual instruction $l$ , with the goal of predicting an action $a_{t}$ , which can be a textual command or a 7-DOF control vector [Δ, Δ. AGrip] depending on the embodiment.  
各タイムステップtにおいて、モデルは視覚的観測 $o_{t}$ とテキストによる指示 $l$ を受け取り、その目標は行動$a_{t}$を予測することです。この行動は、身体化に応じてテキストコマンドまたは7自由度の制御ベクトル[Δ, Δ. AGrip]になります。

To tackle this problem, we propose ThinkAct, a unified framework that aims to leverage an MLLM $\mathcal{F}_{\theta}$ to reason the high-level plans while connecting with an action model $\pi_{\phi}$ to infer executable actions.  
この問題に取り組むために、我々はThinkActを提案します。これは、MLLM $\mathcal{F}_{\theta}$ を活用して高レベルの計画を推論し、実行可能な行動を推測するために行動モデル $\pi_{\phi}$ と接続することを目的とした統一フレームワークです。

The MLLM $\mathcal{F}_{\theta}$ produces a visual plan latent $c_{t}$ based on $(o_{t},l)$ , capturing the high-level intent and planning context (Sec. 3.2).  
MLLM $\mathcal{F}_{\theta}$ は $(o_{t},l)$ に基づいて視覚的な計画の潜在変数 $c_{t}$ を生成し、高レベルの意図と計画の文脈を捉えます (セクション 3.2)。

This reasoned plan $c_{t}$ then guides the downstream action module $\pi_{\phi}$ to sequentially predict N executable actions $[a_{t}]_{t}^{t+N}$ tailored to the target environment (Sec. 3.3).  
この推論された計画 $c_{t}$ は、下流のアクションモジュール $\pi_{\phi}$ を誘導し、対象環境に合わせて調整されたN個の実行可能なアクション $[a_{t}]_{t}^{t+N}$ を順次予測します (セクション 3.3)。

By connecting abstract planning with low-level control, our ThinkAct enables long-horizon reasoning and improves action adaptation in dynamic embodied tasks.  
抽象的な計画と低レベルの制御を結びつけることで、私たちのThinkActは長期的な推論を可能にし、動的な身体化されたタスクにおける行動適応を改善します。

#### **3.2. Reinforced Visual Latent Planning for Embodied Reasoning**

To enable embodied reasoning that generalizes across diverse environments, we aim to incentivize the reasoning capability of multimodal LLMs via reinforcement learning.  
多様な環境にまたがって一般化する身体化された推論を可能にするために、強化学習を通じてマルチモーダルLLMの推論能力を動機付けることを目指します。

A straightforward way is to have the MLLM reason before generating low-level actions, while using the resulting task success rate in target environments as the reward signal.  
簡単な方法は、MLLMに低レベルのアクションを生成する前に推論させ、その結果として得られるターゲット環境でのタスク成功率を報酬信号として使用することです。

However, this approach is restricted to specific simulators without proper guidance from visual scenes.  
しかし、このアプローチは、視覚的なシーンからの適切なガイダンスなしに、特定のシミュレータに限定されます。

##### Reward Shaping from Action-Aligned Visual Feedback <br> 行動に連携した視覚フィードバックからの報酬形成

To tackle this challenge, we design a novel action-aligned visual feedback that captures long-horizon goals and encourages visual grounding during planning.  
この課題に取り組むために、私たちは長期的な目標を捉え、計画中の視覚的な接地を促進する、新しい行動に連携した視覚フィードバックを設計します。

Specifically, inspired by recent works, we are capable of representing high-level plans as spatial-temporal trajectories that capture the gripper end-effector over the visual scene, which serve as a visual-action guidance to steer the embodied reasoning.  
具体的には、最近の研究に触発され、視覚シーン上のグリッパーエンドエフェクタを捉える時空間軌道として高レベルの計画を表現することができ、これが身体化された推論を操縦するための視覚-行動ガイダンスとして機能します。

As depicted in Fig.2(a), given an observation $o_{t}$ at timestep t and a task instruction, the MLLM $\mathcal{F}_{\theta}$ autoregressively generates a sequence of latent embeddings for reasoning $v_{t}\in\mathbb{R}^{|v_{t}|\times d}$ and visual plan $c\in\mathbb{R}^{|c_{t}|\times d}$ where the former is decoded to reasoning steps while the latter would be inferred into a text string of 2D points $\tau=[p_{k}]_{k=1}^{K}$, with $p_{k}\in[0,1]^{2}$, and $p_{3}$ and $p_{K}$ denoting the start and end positions of the gripper.  
図2(a)に示すように、タイムステップtにおける観測$o_{t}$とタスク指示が与えられると、MLLM $\mathcal{F}_{\theta}$は、推論のための潜在埋め込みのシーケンス$v_{t}\in\mathbb{R}^{|v_{t}|\times d}$と視覚プラン$c\in\mathbb{R}^{|c_{t}|\times d}$を自己回帰的に生成します。前者は推論ステップにデコードされ、後者は2D点のテキスト文字列$\tau=[p_{k}]_{k=1}^{K}$（$p_{k}\in[0,1]^{2}$、および$p_{3}$と$p_{K}$はグリッパーの開始位置と終了位置を示す）に推論されます。

As a result, to encourage the model to anticipate visual goal completetion, we introduce the goal reward for comparing predicted start and end positions with corresponding points from trajectory obtained by off-the-shelf detector $\hat{\tau}=[\hat{p}_{k}]_{k=1}^{K}$ as follows,
その結果、モデルが視覚的な目標の完了を予測することを奨励するために、既製の検出器によって得られた軌道$\hat{\tau}=[\hat{p}_{k}]_{k=1}^{K}$からの対応する点と予測された開始位置および終了位置を比較するための目標報酬を次のように導入します。

$r_{goal}=\frac{1}{2}(f(p_{1},\hat{p}_{1})+f(p_{K},\hat{p}_{K}))$ , where $f(p,p^{\prime})=max(0,1-||p-p^{\prime}||_{2}^{2})$
$r_{goal}=\frac{1}{2}(f(p_{1},\hat{p}_{1})+f(p_{K},\hat{p}_{K}))$、ここで$f(p,p^{\prime})=max(0,1-||p-p^{\prime}||_{2}^{2})$

To further enforce the MLLM predicted trajectory to properly correspond to physically plausible gripper motion, the trajectory reward is proposed to regularize the predicted to match the distribution of demonstrated trajectory $\hat{\tau}$.  
MLLMが予測した軌道が物理的に妥当なグリッパーの動きに適切に対応するようにさらに強制するために、予測された軌道が実証された軌道$\hat{\tau}$の分布と一致するように正則化する軌道報酬が提案されています。

Thus, the trajectory reward $r_{traj}$ can be computed as follows,
したがって、軌道報酬$r_{traj}$は次のように計算できます。

$r_{traj}=max(0,1-d(\tau,\hat{\tau}))$

Here, $d(\tau,\hat{\tau})$ denotes a metric measuring the distance between two trajectories, i.e., dynamic time warping (DTW) distance in this work.  
ここで、$d(\tau,\hat{\tau})$は2つの軌道間の距離を測定する指標、すなわちこの研究では動的時間伸縮法（DTW）距離を表します。

The overall reward is thus defined as the combination of our proposed action-aligned visual feedback and the format correctness score $r_{format}$ following existing reasoning works:
したがって、全体的な報酬は、提案された行動に連携した視覚フィードバックと、既存の推論研究に従った形式の正しさのスコア$r_{format}$の組み合わせとして定義されます。

$r=0.9r_{visual}+0.1r_{format}$ where $r_{visual}=\omega_{goal}r_{goal}+\omega_{traj}r_{traj}$
$r=0.9r_{visual}+0.1r_{format}$ ここで $r_{visual}=\omega_{goal}r_{goal}+\omega_{traj}r_{traj}$

Here, $\omega_{goal}=\omega_{traj}=0.5$ are the weighting coefficients for the goal and trajectory rewards.  
ここで、$\omega_{goal}=\omega_{traj}=0.5$は、目標報酬と軌道報酬の重み付け係数です。

##### Reinforced Fine-Tuning for Eliciting Visual Latent Planning <br> 視覚的潜在計画を引き出すための強化学習ファインチューニング

To incentivize the embodied reasoning from the MLLM $\mathcal{F}_{\theta}$, we perform reinforced fin-tuning using Group Relative Policy Optimization (GRPO).  
MLLM $\mathcal{F}_{\theta}$からの身体化された推論を動機付けるために、グループ相対方策最適化（GRPO）を使用して強化学習ファインチューニングを実行します。

Specifically, given an input $(o_{t},l),$ GRPO first samples a group of M distinct responses $\{z_{1},z_{2},...,z_{M}\}$ from the original MLLM $\mathcal{F}_{\theta_{old}}$. Each response is evaluated using the reward function defined in Eq. 3 and resulting in a set of reward signals $\{r_{1},r_{2},...,r_{M}\}$.  
具体的には、入力$(o_{t},l)$が与えられると、GRPOはまず元のMLLM $\mathcal{F}_{\theta_{old}}$ からM個の異なる応答のグループ$\{z_{1},z_{2},...,z_{M}\}$ をサンプリングします。各応答は式3で定義された報酬関数を使用して評価され、報酬信号のセット$\{r_{1},r_{2},...,r_{M}\}$が結果として得られます。

Thus, we optimize $\mathcal{F}_{\theta}$ by maximizing the following objective:
したがって、次の目的を最大化することによって$\mathcal{F}_{\theta}$を最適化します。

$\mathfrak{L}_{GRPO}(\theta)=\frac{1}{M}\sum_{i=1}^{M}(\frac{\mathcal{F}_{\theta}(z_{i}|o_{t},l)}{\mathcal{F}_{\theta_{old}}(z_{i}|o_{t},l)}A_{i}-\beta D_{KL}(\mathcal{F}_{\theta}(z_{i}|o_{t},l)||\mathcal{F}_{\theta_{old}}(z_{i}|o_{t},l)))$

where $A_{i} = \frac{r_i - \text{mean}(\{r_1,...,r_M\})}{\text{std}(\{r_1,...,r_M\})}$

Here, $A_{i}$ quantifies the relative quality of i-th response compared to other candidates in the sampled group.  
ここで、$A_{i}$は、サンプリングされたグループ内の他の候補と比較したi番目の応答の相対的な品質を定量化します。

$D_{KL}(\cdot||\cdot)$ is the KL divergence introduced with a weighting factor to regularize the model, preventing excessive deviation from the original model $\mathcal{F}_{\theta_{old}}$.  
$D_{KL}(\cdot||\cdot)$ は、モデルを正則化するために重み付け係数とともに導入されたKLダイバージェンスであり、元のモデル $\mathcal{F}_{\theta_{old}}$ からの過度の逸脱を防ぎます。

To further obtain general embodied knowledge, our ThinkAct is flexible to encapsulate the publicly available question-answering data to enhance capabilities such as robotic VQA or failure detection by formatting them into the QA-style accuracy reward.  
さらなる一般的な身体知を獲得するために、我々のThinkActは、ロボットVQAや失敗検出などの能力を向上させるために、公開されている質疑応答データをカプセル化し、QA形式の正解率報酬にフォーマットする柔軟性を備えています。

Once the reinforced fine-tuning is complete, we are able to produce long CoT steps, while abstracting the textual reasoning into a compact visual plan latent $c_{t}$, capturing long-horizon spatial-temporal planning intent.  
強化学習によるファインチューニングが完了すると、長いCoTステップを生成できるようになり、同時にテキストによる推論をコンパクトな視覚計画の潜在変数 $c_{t}$ に抽象化して、長期的な時空間計画の意図を捉えることができます。

#### 3.3. Reasoning-Enhanced Action Adaptation

With the high-level embodied intent reasoned by the MLLM, our goal is to connect the inferred visual latent planning $c_t$ with the action model of the target environment in a think-before-acting manner, grounding embodied reasoning into the physical world with executable actions.  
MLLMによって推論された高レベルの身体化された意図を用いて、私たちの目標は、推論された視覚的な潜在計画$c_t$をターゲット環境の行動モデルと「行動する前に考える」方法で結びつけ、身体化された推論を実行可能な行動で物理世界に接地させることです。

Specifically, we build upon a Transformer-based action model $\pi_{\phi}$ (e.g., Diffusion Policy), which predicts actions based on the current state composed of visual observations and language instructions.  
具体的には、視覚的観測と言語指示からなる現在の状態に基づいて行動を予測する、Transformerベースの行動モデル$\pi_{\phi}$（例：拡散方策）を構築します。

While $\pi_{\phi}$ can operate in the target environment using perception alone, we enhance its capability by conditioning it on the latent plan $c_{t},$ which encodes high-level embodied intent and planning context.  
$\pi_{\phi}$は知覚のみを使用してターゲット環境で動作できますが、高レベルの具体化された意図と計画コンテキストをエンコードする潜在計画$c_{t}$で条件付けすることにより、その機能を強化します。

As depicted in Fig.2(b), we incorporate $c_{t}$ using a latent projector to connect it to the input space of the action model, enabling the reasoning guidance to be effectively leveraged, which enhances its low-level action execution in the target environment.  
図2(b)に示すように、潜在プロジェクターを用いて$c_{t}$をアクションモデルの入力空間に接続することで、推論ガイダンスを効果的に活用し、ターゲット環境における低レベルのアクション実行を強化します。

Thus, we solely update the state encoder, latent projector, and action model by imitation learning with annotated action demonstrations:
したがって、注釈付きの行動デモンストレーションを用いた模倣学習によって、状態エンコーダ、潜在プロジェクタ、および行動モデルのみを更新します。

$\mathcal{L}_{IL}(\phi)=\mathbb{E}_{(o_{i},l,a_{i})}[l(\pi_{\phi}(c_{t},o_{i},l),a_{i})]$.  

We note that, reasoning and action execution could be operated in an asynchronous manner, which means each latent plan $c_{t}$ corresponds to N interactions with the environment (i.e., $i\in[t,t+N]$).  
推論と行動の実行は非同期的に操作できることに注意してください。これは、各潜在計画$c_{t}$が環境とのN回の相互作用（つまり、$i\in[t,t+N]$）に対応することを意味します。

This asynchronous design highlights a key advantage of our dual-system architecture, allowing the reasoning MLLM to perform slow thinking while the action model executes fast control.  
この非同期設計は、私たちのデュアルシステムアーキテクチャの重要な利点を浮き彫りにします。これにより、推論MLLMがゆっくりとした思考を実行する一方で、アクションモデルが高速な制御を実行できます。

#### 3.4. Learning Strategy and Inference

Following Feng et al. (2025), we adopt a multi-stage training strategy for our ThinkAct.  
Feng et al. (2025) に倣い、ThinkActでは多段階のトレーニング戦略を採用しています。

Before RL, we initialize the two modules independently.  
RLの前に、2つのモジュールを独立して初期化します。

The MLLM $\mathcal{F}_{\theta}$ is cold-started using supervised data (Sec. 4.1) to learn to interpret visual trajectories and produce reasoning and answers in the correct output format.  
MLLM $\mathcal{F}_{\theta}$は、教師ありデータ（セクション4.1）を使用してコールドスタートされ、視覚的な軌道を解釈し、正しい出力形式で推論と回答を生成することを学習します。

On the other hand, the action model $\pi_{\phi}$ is pre-trained on the Open X-Embodiment (OXE) dataset, providing a strong foundation for low-level action execution.  
一方、行動モデル $\pi_{\phi}$ はOpen X-Embodiment（OXE）データセットで事前学習されており、低レベルの行動実行のための強力な基盤を提供します。

After SFT cold-start, our MLLM $\mathcal{F}_{\theta}$ is tuned with action-aligned rewards guiding the generation of effective latent plans.  
SFTコールドスタート後、我々のMLLM $\mathcal{F}_{\theta}$は、効果的な潜在プランの生成を導く行動に連携した報酬で調整されます。

During reasoning-enhanced action adaptation, we freeze $\mathcal{F}_{\theta}$ while updating the action model $\pi_{\phi}$ with state encoder and latent projector on the target environment by conditioning on the latent visual plan $c_{t}$.  
推論強化行動適応中、我々は潜在視覚計画$c_{t}$を条件として、対象環境上の状態エンコーダと潜在プロジェクタで行動モデル$\pi_{\phi}$を更新しながら、$\mathcal{F}_{\theta}$を凍結します。

At inference time, given a visual observation $o_{t}$ and instruction $l$, ThinkAct produces a visual plan latent $c_{t}=\mathcal{F}_{\theta}(o_{t},l)$, which conditions the action module $\pi_{\phi}$ to predict a sequence of executable actions tailored to the current environment.  
推論時、視覚的観測$o_{t}$と指示$l$が与えられると、ThinkActは視覚的計画の潜在変数$c_{t}=\mathcal{F}_{\theta}(o_{t},l)$を生成します。これは、現在の環境に合わせて調整された一連の実行可能な行動を予測するように行動モジュール$\pi_{\phi}$を条件付けます。

### **4\. Experiment**

#### **4.1. Experimental Setup**

##### Implementation Details

We initialize $\mathcal{F}_{\theta}$ with Qwen2.5-VL 7B.  
$\mathcal{F}_{\theta}$ をQwen2.5-VL 7Bで初期化します。

The cold-start stage runs for 20K iterations with batch size 32 and learning rate 1e-5 using DeepSpeed ZeRO-3.  
コールドスタート段階は、DeepSpeed ZeRO-3を使用し、バッチサイズ32、学習率1e-5で20Kイテレーション実行されます。

We then apply GRPO for 6K iterations, using batch size 64, learning rate 1e-6, and rollout size 5.  
次に、GRPOをバッチサイズ64、学習率1e-6、ロールアウトサイズ5で6K回適用します。

The action model $\pi_{\phi}$ is a DiT-based policy with 432M parameters, pre-trained using the OXE dataset, where the state encoder is composed of a DINOv2 image encoder and a CLIP text encoder that jointly encode the current state inputs into 1024-dim embeddings.  
行動モデル $\pi_{\phi}$ は、4億3200万個のパラメータを持つDiTベースの方策であり、OXEデータセットを使用して事前学習されています。状態エンコーダは、DINOv2画像エンコーダとCLIPテキストエンコーダで構成され、現在の状態入力を共同で1024次元の埋め込みにエンコードします。

For reasoning-enhanced action adaptation, we connect the visual plan $c_{t}$ via a Q-Former as the latent projector with 32 queries and fine-tune on 100K OXE samples for 120K iterations using batch size 256 and learning rate 2e-5.  
推論強化行動適応のために、我々は視覚計画 $c_{t}$ を、32個のクエリを持つ潜在プロジェクタとしてQ-Formerを介して接続し、バッチサイズ256と学習率2e-5を使用して100K個のOXEサンプルで120K回ファインチューニングします。

LIBERO tasks are further fine-tuned for 75K iterations with batch size 128.  
LIBEROタスクは、バッチサイズ128でさらに75,000回ファインチューニングされます。

All experiments are conducted on 16 NVIDIA A100 GPUs with 80 GB memory.  
すべての実験は、80GBのメモリを搭載した16台のNVIDIA A100 GPUで実施されました。

##### **Training Datasets and Evaluation Benchmarks**

For SFT cold-start, we fine-tune the MLLM using trajectories from the subset of OXE, and QA tasks from RoboVQA, EgoPlan-IT, and Video-R1-CoT.  
SFTコールドスタートでは、OXEのサブセットからの軌道、RoboVQA、EgoPlan-IT、Video-R1-CoTからのQAタスクを使用してMLLMをファインチューニングします。

During RL training, we incorporate trajectories from the OXE subset and human videos from Something-Something v2.  
RLトレーニング中、OXEサブセットからの軌道とSomething-Something v2からのヒューマンビデオを組み込みます。

To enhance general reasoning capability, we include embodied QA datasets such as EgoPlan-IT/Val, RoboVQA, and the Reflect dataset, as well as a general video instruction dataset, i.e., LLaVA-Video-178K.  
一般的な推論能力を強化するために、EgoPlan-IT/Val、RoboVQA、Reflectデータセットなどの具体化されたQAデータセット、および一般的なビデオ指示データセット、つまり LLaVA-Video-178K を含めます。

We evaluate ThinkAct on two robot manipulation and three embodied reasoning benchmarks.  
ThinkActを2つのロボット操作と3つの身体化推論ベンチマークで評価します。

For manipulation tasks, SimplerEnv containing diverse scenes and LIBERO with long-horizon tasks are evaluated using task success rate.  
操作タスクについては、多様なシーンを含むSimplerEnvと長期タスクを持つLIBEROをタスク成功率を用いて評価します。

For reasoning benchmarks, EgoPlan-Bench2 uses accuracy on multiple-choice questions, while RoboVQA and OpenEQA are free-form QA tasks evaluated using BLEU score and LLM-based scoring, respectively, following their original protocols.  
推論ベンチマークでは、EgoPlan-Bench2は多肢選択問題の正解率を使用し、RoboVQAとOpenEQAは自由形式のQAタスクで、それぞれ元のプロトコルに従ってBLEUスコアとLLMベースのスコアリングを使用して評価されます。

Further details of our experimental setup are provided in the supplementary material.  
実験設定の詳細は、補足資料に記載されています。

#### **4.2. Quantitative Evaluation**

##### **Robot Manipulation**

To assess the effectiveness of ThinkAct on robot manipulation task, we evaluate on SimplerEnv and LIBERO.  
ロボット操作タスクにおけるThinkActの有効性を評価するために、SimplerEnvとLIBEROで評価します。

SimplerEnv includes Google-VM (Visual Matching), Google-VA (Variant Aggregation), and Bridge-VM setups, introducing variations in color, material, lighting, and camera pose to evaluate model robustness.  
SimplerEnvには、Google-VM（視覚的マッチング）、Google-VA（バリアント集約）、およびBridge-VMのセットアップが含まれており、モデルの堅牢性を評価するために、色、素材、照明、カメラのポーズにバリエーションを導入しています。

For the LIBERO benchmark, following prior works, we evaluate on the LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, and LIBERO-Long subtasks to test model generalization across spatial layouts, object variations, goal diversity, and long-horizon planning.  
LIBEROベンチマークについては、先行研究に倣い、LIBERO-Spatial、LIBERO-Object、LIBERO-Goal、LIBERO-Longのサブタスクで評価し、空間レイアウト、オブジェクトのバリエーション、目標の多様性、長期計画にわたるモデルの汎化性能をテストします。

As shown in Tab. 1, on the SimplerEnv, incorporating our reasoning-guided visual plan latents allows ThinkAct to outperform our baseline action model, DiT-Policy, by 15.5%, 16.9%, and 11.4% on Google-VM, Google-VA, and Bridge-VM, respectively, achieving the highest overall scores of 71.5%, 65.1%, and 43.8% against all methods.  
表1に示すように、SimplerEnvにおいて、我々の推論誘導視覚計画潜在変数を組み込むことにより、ThinkActはベースライン行動モデルであるDiT-PolicyをGoogle-VM、Google-VA、Bridge-VMでそれぞれ15.5%、16.9%、11.4%上回り、全手法に対して71.5%、65.1%、43.8%という最高の総合スコアを達成しました。

On the LIBERO benchmark, ThinkAct achieves the best overall success rate of 84.4%, outperforming DiT-Policy and recent state-of-the-art CoT-VLA, verifying the effectiveness on diverse robotic manipulation settings.  
LIBEROベンチマークにおいて、ThinkActは84.4%という最高の総合成功率を達成し、DiT-Policyや最近の最先端技術であるCoT-VLAを上回り、多様なロボット操作設定における有効性を検証しました。

##### **Embodied Reasoning**

In Tab. 2, we assess the reasoning capability of ThinkAct in embodied scenarios on three benchmarks: EgoPlan-Bench2, RoboVQA, and OpenEQA.  
表2では、EgoPlan-Bench2、RoboVQA、OpenEQAの3つのベンチマークを用いて、身体化シナリオにおけるThinkActの推論能力を評価します。

EgoPlan-Bench2 measures multi-step planning in egocentric daily-life scenarios, while RoboVQA focuses on long-horizon reasoning in robotic manipulation.  
EgoPlan-Bench2は、自己中心的な日常生活シナリオにおける複数ステップの計画を測定するのに対し、RoboVQAはロボット操作における長期的な推論に焦点を当てています。

ThinkAct outperforms the second-best method by 2.5% and 4.1 BLEU score on these two benchmarks, demonstrating its strength in long-horizon and multi-step planning.  
ThinkActは、これら2つのベンチマークで2番目に優れた手法を2.5%と4.1 BLEUスコア上回り、長期および複数ステップの計画における強みを示しています。

Separately, OpenEQA measures zero-shot embodied understanding across diverse environments.  
別途、OpenEQAは多様な環境におけるゼロショットの身体化理解を測定します。

The enhanced reasoning ability of ThinkAct enables better generalization and scene comprehension, resulting in strong performance on this benchmark.  
ThinkActの強化された推論能力は、より優れた一般化とシーン理解を可能にし、このベンチマークで高いパフォーマンスをもたらします。

#### **4.3. Qualitative Results**

In Fig.3, we qualitatively showcase the reasoning process and execution scenes of two manipulation examples from the Simpler-Bridge and LIBERO-Long tasks.  
図3では、Simpler-BridgeとLIBERO-Longタスクからの2つの操作例の推論プロセスと実行シーンを定性的に示します。

In the LIBERO-Long task "Pick up the book and place it in the back compartment," ThinkAct decomposes the instruction into sub-tasks: (1) pick up the book, (2) move from left to right, and (3) place it in the compartment, demonstrating its long-horizon planning capability.  
LIBERO-Longタスク「本を拾って後ろの区画に置く」において、ThinkActは指示をサブタスクに分解します：
（1）本を拾う、（2）左から右へ移動する、（3）区画に置く。
これは、その長期計画能力を示しています。

We also visualize the planned trajectory, confirming that the gripper closely follows the reasoning-guided plan during execution.  
また、計画された軌道を可視化し、実行中にグリッパーが推論によって導かれた計画に厳密に従っていることを確認します。

To better illustrate the impact of RL on the reasoning process, Fig.4 compares ThinkAct before and after RL fine-tuning on embodied reasoning tasks.  
RLが推論プロセスに与える影響をよりよく説明するために、図4では、身体化推論タスクにおけるRLファインチューニング前後のThinkActを比較しています。

As we can observe in Fig.4(a), using a RoboVQA example, the SFT cold-start model focuses only on the current state and fails to reason over future steps, while the RL-tuned model successfully infers the correct answer.  
図4(a)のRoboVQAの例を見ると、SFTコールドスタートモデルは現在の状態にのみ焦点を当て、将来のステップについて推論することに失敗していますが、RLで調整されたモデルは正解を正しく推測しています。

Also, as demonstrated in Fig.4(b), from OpenEQA, the cold-start model misinterprets the question, whereas the RL-tuned version demonstrates improved question and environment understanding.  
また、図4(b)のOpenEQAの例で示されているように、コールドスタートモデルは質問を誤解していますが、RLで調整されたバージョンは質問と環境の理解が向上していることを示しています。

More qualitative comparisons and demo videos are provided in the supplementary material.  
より定性的な比較とデモビデオは補足資料で提供されています。

#### 4.4. Ablation Study

In Tab. 3, we ablate the proposed goal reward $r_{goal}$ and trajectory reward $r_{traj}$ to analyze their individual contributions to reasoning and planning.  
表3では、提案された目標報酬 $r_{goal}$ と軌道報酬 $r_{traj}$ を除去し、推論と計画への個々の貢献を分析します。

We start from the full version of ThinkAct, which achieves the best performance across all benchmarks.  
すべてのベンチマークで最高のパフォーマンスを達成するThinkActの完全版から始めます。

Removing the trajectory reward leads to a noticeable drop, indicating that $r_{traj}$ is essential for learning coherent and structured planning behaviors.  
軌道報酬を削除すると顕著な低下が見られ、$r_{traj}$が首尾一貫した構造化された計画行動を学習するために不可欠であることが示唆されます。

Without the goal reward, performance also declines, suggesting that $r_{goal}$ plays a key role in incentivizing long-horizon reasoning.  
目標報酬がなければ、パフォーマンスも低下し、$r_{goal}$が長期的な推論を動機付ける上で重要な役割を果たすことを示唆しています。

When both $r_{traj}$ and $r_{goal}$ are removed, leaving only QA-style reward from QA datasets, the model shows only marginal improvements over the SFT baseline, confirming that action-aligned visual feedback is critical for effective multi-step planning in embodied settings.  
$r_{traj}$ と $r_{goal}$ の両方が削除され、QAデータセットからのQAスタイルの報酬のみが残された場合、モデルはSFTベースラインに対してわずかな改善しか示さず、身体化された設定における効果的な複数ステップの計画には、行動に連携した視覚フィードバックが不可欠であることを裏付けています。

Finally, the SFT cold-start model without RL yields the lowest scores, verifying the effectiveness of our RL fine-tuning for eliciting the reasoning capability in MLLMs.  
最後に、RLなしのSFTコールドスタートモデルは最も低いスコアとなり、MLLMにおける推論能力を引き出すための我々のRLファインチューニングの有効性を検証しています。

More ablation studies (e.g., the number of interactions per reasoning step N) are provided in the supplementary material.  
より多くの除去研究（例えば、推論ステップごとの相互作用の数N）は、補足資料で提供されています。

#### **4.5. Analysis of ThinkAct**

In this section, we analyze the capabilities of ThinkAct in enhancing robotic manipulation by embodied reasoning.  
本セクションでは、身体化された推論によってロボット操作を強化するThinkActの能力を分析します。

We focus on two key aspects: (1) how reasoning facilitates effective few-shot adaptation to new tasks and environments, and (2) how it enables the robot to detect failures and perform self-correction during task execution.  
我々は2つの重要な側面に焦点を当てます：(1) 推論が新しいタスクや環境への効果的な few-shot 適応をどのように促進するか、(2) タスク実行中にロボットが失敗を検出し自己修正することをどのように可能にするか。

Through both quantitative experiments and qualitative examples, we demonstrate the unique advantages of leveraging a reasoning MLLM to tackle embodied action tasks.  
定量的実験と定性的事例の両方を通じて、身体化された行動タスクに取り組むために推論MLLMを活用することのユニークな利点を実証します。

We further provide the analysis of MLLM backbones in the supplementary material.  
補足資料では、MLLMバックボーンの分析をさらに提供しています。

##### **Reasoning Enhance Few-Shot Adaptation** <br> **推論による few-shot 適応の強化**

As we can observe in Fig.3 and Fig.4, ThinkAct is capable of describing the environment and decomposing task instructions into meaningful sub-goals.  
図3と図4で観察できるように、ThinkActは環境を記述し、タスクの指示を意味のあるサブゴールに分解することができます。

To validate whether such reasoning improves the action model's adaptability, we conduct a few-shot adaptation experiment on the LIBERO benchmark.  
このような推論が行動モデルの適応性を向上させるかどうかを検証するために、LIBEROベンチマークで few-shot 適応実験を行います。

Specifically, we use LIBERO-Spatial and LIBERO-Object to evaluate adaptation to unseen environments, and LIBERO-Goal to test adaptation to new skills.  
具体的には、LIBERO-SpatialとLIBERO-Objectを使用して未知の環境への適応を評価し、LIBERO-Goalを使用して新しいスキルへの適応をテストします。

We fine-tune the action model on just 10 demonstrations per task and evaluate performance over 100 trials.  
タスクごとにわずか10回のデモンストレーションで行動モデルを微調整し、100回の試行でパフォーマンスを評価します。

As shown in Fig.5, ThinkAct consistently outperforms state-of-the-art methods, achieving the highest success rates across all tasks.  
図5に示すように、ThinkActは一貫して最先端の手法を上回り、すべてのタスクで最高の成功率を達成しています。

Notably, it surpasses Magma by 7.3% on LIBERO-Goal and by 9.5% on LIBERO-Spatial, demonstrating the effectiveness of reasoning capability for few-shot generalization in both novel skills and environments.  
特に、LIBERO-GoalでMagmaを7.3%、LIBERO-Spatialで9.5%上回り、新しいスキルと環境の両方における few-shot 汎化に対する推論能力の有効性を示しています。

##### Reasoning Elicit Self-Correction <br> 自己修正を引き出す推論

Failure detection and self-correction are critical for robust robot manipulation.  
故障検知と自己修正は、堅牢なロボット操作にとって極めて重要です。

To evaluate whether ThinkAct can reason about and recover from execution errors, we enable the reasoning MLLM to observe more contextual information during execution by extending its input from a single image $O_{t}$ to a short video segment $o_{t-N:t}$ .  
ThinkActが実行エラーについて推論し、回復できるかどうかを評価するために、推論MLLMの入力を単一の画像 $O_{t}$ から短いビデオセグメント $o_{t-N:t}$ に拡張することで、実行中により多くの文脈情報を観察できるようにします。

This temporal context allows ThinkAct to detect failures, reconsider the situation, and replan accordingly.  
この時間的文脈により、ThinkActは失敗を検出し、状況を再検討し、それに応じて再計画することができます。

For example, as shown in Fig.6, in a task where the robot is instructed to place a box into a basket, the gripper accidentally drops the box midway.  
例えば、図6に示すように、ロボットに箱をかごに入れるように指示されたタスクで、グリッパーが途中で誤って箱を落としてしまいます。

The reasoning MLLM identifies the failure, says "Let's reconsider how to complete the task," and generates a revised plan that guides the gripper back to the dropped location to regrasp the box.  
推論MLLMは失敗を特定し、「タスクを完了する方法を再検討しましょう」と言い、グリッパーを落とした場所に戻して箱を再びつかむように導く修正計画を生成します。

The robot then successfully completes the task, demonstrating ThinkAct's ability to reflect on errors and self-correct through structured reasoning.  
その後、ロボットはタスクを正常に完了し、構造化された推論を通じてエラーを反省し、自己修正するThinkActの能力を実証します。

### **5\. Conclusion**

We presented ThinkAct, a framework that reinforces visual latent planning for vision-language-action reasoning tasks.  
我々は、視覚言語行動推論タスクのための視覚的潜在計画を強化するフレームワークであるThinkActを発表しました。

By combining action-aligned reinforcement learning with reasoning-enhanced action adaptation, ThinkAct enables embodied agents to think before acting and execute robust actions in dynamic environments.  
行動に連携した強化学習と推論で強化された行動適応を組み合わせることで、ThinkActは身体化されたエージェントが行動する前に考え、動的な環境で堅牢な行動を実行できるようにします。

Through extensive experiments across embodied reasoning and robot manipulation benchmarks, we demonstrated strong long-horizon planning, few-shot adaptation, and emergent behaviors such as failure detection and self-correction, providing a scalable path toward more deliberative and adaptable embodied Al systems.  
身体化された推論とロボット操作のベンチマークにわたる広範な実験を通じて、我々は強力な長期計画、 few-shot 適応、および故障検出や自己修正などの創発的行動を実証し、より審議的で適応性のある身体化AIシステムに向けたスケーラブルな道筋を提供しました。

#### **Limitations**

Since ThinkAct builds on pretrained multimodal LLMs, it inevitably inherits their limitations, particularly hallucinations in visual or spatial reasoning.  
ThinkActは事前学習済みのマルチモーダルLLMを基盤としているため、特に視覚的または空間的推論における幻覚など、その限界を必然的に受け継ぎます。

This can lead to generated plans that reference incorrect object attributes or spatial relationships, affecting downstream execution.  
これにより、誤ったオブジェクト属性や空間関係を参照する計画が生成され、下流の実行に影響を与える可能性があります。

While our latent planning and action grounding mitigate this to some extent, future work on grounding-aware training or hallucination suppression in MLLMs may further improve robustness and reliability in real-world deployment.  
我々の潜在的計画と行動接地はこれをある程度緩和しますが、MLLMにおける接地を意識した訓練や幻覚抑制に関する今後の研究は、実世界での展開における堅牢性と信頼性をさらに向上させる可能性があります。

#### **Broader Impacts**

Our work aims to enhance the reasoning capabilities of embodied agents, which could support real-world applications such as assistive robotics, home automation, and industrial systems.  
私たちの研究は、支援ロボット、ホームオートメーション、産業システムなどの実世界アプリケーションをサポートできる、身体化されたエージェントの推論能力を向上させることを目的としています。

In particular, models like ThinkAct may help robots better interpret vague instructions and execute multi-step plans in dynamic environments.  
特に、ThinkActのようなモデルは、ロボットが曖昧な指示をよりよく解釈し、動的な環境で複数ステップの計画を実行するのに役立つ可能性があります。

However, increased autonomy and reasoning ability in embodied systems also raise potential concerns.  
しかし、身体化されたシステムにおける自律性と推論能力の向上は、潜在的な懸念も引き起こします。

Misinterpretation of ambiguous commands, reliance on hallucinated visual reasoning, or overconfidence in CoT outputs could result in unintended behaviors, especially in safety-critical settings.  
曖昧なコマンドの誤解、幻覚的な視覚的推論への依存、またはCoT出力への過信は、特に安全性が重要な設定において、意図しない行動につながる可能性があります。

Hence, future research on safeguards or alignment with human intent could further help mitigate these risks.  
したがって、安全策や人間の意図との整合に関する今後の研究は、これらのリスクをさらに軽減するのに役立つ可能性があります。
