# Apresentando o Framework

**Introdução**

Este documento analisa os temas principais e as ideias mais importantes apresentadas em um artigo de pesquisa intitulado "Deep-MDS framework for recovering the 3D shape of 2D landmarks from a single image". O artigo propõe um novo framework chamado DeepMDS para recuperar a forma 3D de marcos faciais 2D a partir de uma única imagem, utilizando uma abordagem baseada em aprendizado profundo e Escala Multidimensional (MDS).

**Principais Temas:**

\- Reconstrução Facial 3D: O artigo aborda o problema da reconstrução facial 3D a partir de uma única imagem, destacando a importância dos marcos geométricos, como pontos de referência, para simplificar o processo e reduzir os recursos computacionais.  
\- Aprendizado Profundo para Reconstrução 3D: O DeepMDS utiliza uma rede neural profunda para aprender a dissimilaridade 3D entre os marcos 2D. Essa rede é treinada para estimar a distância euclidiana 3D entre os marcos correspondentes no espaço 3D.  
\- Abordagem MDS: O MDS é uma técnica de redução de dimensionalidade que preserva as distâncias entre os pontos de dados. No DeepMDS, o MDS é usado para mapear os marcos 2D para o espaço 3D, preservando suas relações espaciais.

**Ideias e Fatos Importantes:**

\- Eficiência Computacional: O uso de marcos em vez da imagem completa reduz significativamente a complexidade computacional, tornando o DeepMDS adequado para dispositivos com recursos limitados.  
\- Estrutura DeepMDS: O framework consiste em três componentes principais:  
Um autoencoder que transforma qualquer visão de entrada em uma visão de perfil.  
Uma rede neural profunda que aprende a dissimilaridade entre pares de marcos 2D.  
O componente MDS que utiliza a matriz de dissimilaridade para recuperar a forma 3D dos marcos.  
\- Vantagens do DeepMDS: O DeepMDS oferece uma solução imparcial para reconstrução 3D, sem depender de um modelo 3D específico. Além disso, é independente do tipo de projeção 2D e apresenta um número reduzido de parâmetros de treinamento em comparação com outros métodos.  
\- Desafios e Limitações: A precisão do DeepMDS depende da qualidade dos marcos detectados na imagem. A complexidade computacional aumenta com o número de marcos, devido à etapa de decomposição da matriz no MDS.  
\- Resultados Experimentais: O DeepMDS foi avaliado em vários conjuntos de dados de faces humanas, demonstrando desempenho superior em termos de precisão e eficiência em comparação com métodos de última geração.

**Citações Relevantes:**

"Utilizing the 3D face shape in a framework helps increase the framework accuracy and makes it invariant to the pose and/or occlusion changes in the input."  
"Landmarks are well-known and efficient features which are applicable in many computer vision tasks \[...\] The main advantage of landmark-based approaches is that they are less sensitive to lightning conditions or other radiometric variations."  
"Deep learning frameworks are highly valuable tools with a wide range of applications in various computer vision domains \[...\] thanks to their ability to learn from data and effectively handle unstructured information."  
"One of the primary limitations \[of existing methods\] is their dependency on a 3D model, which results in solutions that are biased toward the average shape of the associated 3D model."  
"\[Our proposed method\] could be incorporated as the landmark depth estimation component in these methods to increase their efficiency."

**Conclusão:**

O DeepMDS é um framework que utiliza uma combinação de aprendizado profundo e MDS para reconstruir a forma 3D de marcos faciais 2D a partir de uma única imagem. O framework é eficiente em termos de recursos computacionais, oferece soluções imparciais e apresenta alta precisão em diversos conjuntos de dados.   

# Vantagens do Framework

### **1\. Objetivo do Framework**

O DeepMDS foi projetado para reconstruir a localização 3D de marcos 2D do rosto humano a partir de uma única imagem, independentemente do tipo de formação da imagem, como pose ou tipo de projeção. Isso é particularmente desafiador, pois a recuperação de formas 3D a partir de imagens 2D é um problema mal posicionado que requer a imposição de restrições para obter soluções viáveis.

### **2\. Componentes do Framework**

O framework combina duas abordagens principais:

* **Multi-Dimensional Scaling (MDS)**: O MDS é utilizado para mapear os pontos 2D em um espaço de forma 3D, preservando a configuração dos marcos. Ele aprende a distância euclidiana 3D a partir das distâncias dos marcos 2D, resultando em uma dissimilaridade simétrica na abordagem não métrica do MDS. Isso permite que o framework evite soluções enviesadas em relação a um modelo 3D específico, proporcionando uma representação mais flexível e precisa das variações emocionais e expressões faciais.

* **Deep Learning**: O framework incorpora componentes de aprendizado profundo para aprender a dissimilaridade entre pares de pontos 2D em uma imagem. A abordagem de aprendizado profundo é projetada para ter um número reduzido de parâmetros treináveis, o que a torna eficiente em termos de computação.

### **3\. Vantagens do DeepMDS**

* **Independência de Modelos 3D Pré-definidos**: Ao não depender de um modelo 3D específico, o DeepMDS evita o viés que pode ocorrer em métodos tradicionais, onde as soluções tendem a se aproximar da forma média do modelo utilizado.

* **Flexibilidade em Relação a Diferentes Projeções**: O framework é capaz de lidar com diferentes tipos de projeções e poses, o que é crucial para a aplicação em cenários do mundo real, onde as imagens podem variar amplamente.

* **Eficiência Computacional**: Embora o MDS envolva uma etapa de decomposição de matriz que pode aumentar a complexidade computacional com um grande número de marcos, o framework busca mitigar isso utilizando aproximações computacionais de baixo custo para o processo de decomposição de autovalores.

### **4\. Limitações e Futuras Direções**

O artigo também discute algumas limitações do framework, como a necessidade de calibração dos marcos de entrada para identificar padrões, especialmente em visões de perfil. Para abordar isso, os autores sugerem o desenvolvimento de estruturas de proximidade mais complexas, combinando múltiplas redes de proximidade.

Além disso, futuras pesquisas podem se concentrar na aplicação do método a objetos além do rosto humano, resolvendo fenômenos de auto-oclusão e investigando soluções para as limitações existentes, como a criação de uma estrutura de dissimilaridade mais complexa.

### **5\. Resultados e Conclusões**

Os resultados experimentais demonstram que o DeepMDS é eficaz na recuperação de formas de marcos 3D, apresentando uma avaliação de desempenho que valida sua plausibilidade em relação a métodos semelhantes. O framework pode ser integrado em sistemas de reconstrução 3D de alta resolução ou em dispositivos móveis, oferecendo uma solução de baixo parâmetro com precisão promissora.

Em resumo, o DeepMDS representa um avanço significativo na área de reconstrução 3D, oferecendo uma abordagem inovadora e eficiente para lidar com os desafios da recuperação de formas a partir de imagens 2D.

# Comparativo com o Mediapipe

### **1\. Abordagem de Recuperação de Forma 3D**

* **DeepMDS**: O framework utiliza uma combinação de Multi-Dimensional Scaling (MDS) e aprendizado profundo para mapear marcos 2D em um espaço 3D. Ele aprende a dissimilaridade entre pares de pontos 2D e utiliza essa informação para reconstruir a forma 3D de maneira independente de modelos 3D pré-definidos, evitando viés em relação a uma forma média.

* **Mediapipe**: O Mediapipe é uma biblioteca de código aberto que fornece detecção de marcos faciais em 3D. Ele utiliza uma abordagem baseada em aprendizado de máquina para detectar 478 pontos 3D em um rosto humano a partir de uma única imagem. O Mediapipe é mais focado na detecção de pontos e na aplicação em tempo real, sendo uma solução prática para várias aplicações de percepção.

### **2\. Dependência de Modelos 3D**

* **DeepMDS**: O framework é projetado para ser independente de um modelo 3D específico, o que significa que ele não se baseia em uma forma média ou em um modelo pré-definido. Isso permite uma maior flexibilidade e precisão na recuperação de formas 3D, especialmente em diferentes poses e expressões faciais.

* **Mediapipe**: Embora o Mediapipe possa detectar marcos 3D, ele pode estar mais limitado em termos de flexibilidade em relação a diferentes expressões faciais e poses, uma vez que sua abordagem pode ser mais orientada por um modelo de referência.

### **3\. Complexidade Computacional**

* **DeepMDS**: O framework busca ser eficiente em termos de computação, utilizando um número reduzido de parâmetros treináveis. No entanto, a etapa de decomposição de matriz no MDS pode aumentar a complexidade computacional, especialmente com um grande número de marcos.

* **Mediapipe**: O Mediapipe é otimizado para aplicações em tempo real e é projetado para ser leve e rápido, permitindo a detecção de marcos em dispositivos móveis e em ambientes de execução rápida.

### **4\. Resultados e Avaliação de Desempenho**

* **DeepMDS**: Os resultados experimentais do DeepMDS demonstram sua eficácia na recuperação de formas 3D de marcos 2D, apresentando uma avaliação de desempenho que valida sua abordagem em comparação com métodos existentes.

* **Mediapipe**: O Mediapipe é amplamente utilizado e reconhecido por sua capacidade de detectar marcos faciais em tempo real, mas o artigo menciona que, em alguns casos, ele pode não encontrar soluções para a recuperação de formas 3D, conforme indicado pela notação "N/A" em alguns resultados.

### **5\. Aplicações e Integração**

* **DeepMDS**: O framework pode ser integrado em sistemas de reconstrução 3D de alta resolução ou em dispositivos móveis, oferecendo uma solução de baixo parâmetro com precisão promissora.

* **Mediapipe**: É uma ferramenta prática para aplicações de percepção em tempo real, como realidade aumentada e rastreamento facial, sendo amplamente utilizada em várias aplicações comerciais e de pesquisa.

### **Conclusão**

Em resumo, enquanto o DeepMDS se concentra na recuperação precisa de formas 3D a partir de marcos 2D de maneira independente de modelos, o Mediapipe é uma solução prática e otimizada para detecção de marcos faciais em tempo real. 

Em termos de viabilidade, fazer a troca do Mediapipe nos atuais projetos pelo DeepMDS não haveria tantos ganhos em comparação com o tempo e energia gastos na transição. Para as finalidades desejadas atualmente, o Mediapipe e o DeepMDS cumprem basicamente a mesma função, existem algumas diferenças de desempenho, que, numericamente, são extremamente pequenas (na ordem da oitava casa decimal).

Um ponto forte em que o DeepMDS se sobressai é na detecção de pontos da face quando a pose está angulada (rosto em perfil). Neste estado, o Mediapipe tem dificuldades em encontrar os pontos, já o DeepMDS consegue executar a tarefa com facilidade e precisão. Porém, mesmo com esta vantagem, nossas análises não ocorrem com o usuário com o rosto angulado, então não se justifica a troca.

