# Plano de Refatoração para UI e Pipeline do Classificador Raster

## 1. Objetivo

Integrar a lógica do `main6_multcore.py` diretamente na interface de usuário (`ui_main.py` + `core/main_controller.py`), separando responsabilidades em classes definidas para facilitar manutenção, testes e evolução.

## 2. Principais componentes identificados

### 2.1 UI
- `MainWindow` (`ui_main.py`)
  - Responsável pela construção da interface gráfica.
  - Deve conter apenas a definição dos widgets, layout e estilos.
  - Não deve implementar lógica de pipeline nem processamento.

- `PathBrowseRow` (`ui_main.py`)
  - Componente de entrada de caminho de arquivo.
  - Deve manter apenas comportamento de seleção de arquivo/diretório e leitura de texto.

- `MainController` (`core/main_controller.py`)
  - Responsável por ligar os sinais da UI e orquestrar ações.
  - Deve ser a ponte para executar o pipeline, validar entradas, atualizar o resumo e o log.
  - Deve delegar toda a lógica de processamento para classes de domínio ou serviços.

### 2.2 Pipeline / Domínio
- `PipelineConfig`
  - Carrega e valida a configuração da UI.
  - Converte os valores de interface em um objeto fortemente tipado.
  - Resolve caminhos relativos e garante consistência de parâmetros.

- `HardwareManager`
  - Avalia RAM e dispositivos TensorFlow / CPU.
  - Calcula chunk size e limites de memória.
  - Fornece dados de hardware para exibição e controle de desempenho.

- `RasterSource`
  - Encapsula leitura de raster de treino e classificação.
  - Fornece metadados e função `sample(coords)`.
  - Controla abertura/fechamento correto do recurso.

- `ShapefileDataset`
  - Carrega shapefiles de amostra e unifica CRS.
  - Mapeia classes, nomes e fornece coordenadas de pontos.

- `FeatureExtractor`
  - Extrai espectros das coordenadas do shapefile no raster de treino.
  - Aplica máscara quando necessário.

- `DatasetSplitter`
  - Separa treino/teste com estratificação.
  - Calcula contagens de cada classe.

- `ModelFactory`
  - Cria a arquitetura Keras com base na configuração.
  - Decide função de perda e ativação de saída.

- `Trainer`
  - Executa o ajuste do modelo.
  - Salva o modelo se solicitado.

- `Evaluator`
  - Avalia desempenho no conjunto de teste.
  - Gera métricas, relatório, gráficos e matriz de confusão.

- `RasterPredictor`
  - Classifica a imagem em chunks.
  - Escreve GeoTIFF de saída com compressão, nodata e overviews.
  - Atualiza progresso e status.

- `Pipeline` / `ClassifierPipeline`
  - Orquestra todos os componentes acima.
  - Executa cada etapa em sequência.
  - Expõe métodos para integração com a UI.

## 3. Atualização do planejamento para UI

### 3.1 Revisão do `MainWindow`

**Fazer**
- Manter apenas widgets, layout e estilos.
- Definir IDs ou atributos públicos para acesso do controlador.
- Incluir área de log, barra de progresso e resumo configurável.

**Não fazer**
- Não deve ter lógica de pipeline ou validações complexas.
- Não deve acessar arquivos ou treinar modelo diretamente.

### 3.2 Revisão do `MainController`

**Fazer**
- Centralizar a gestão de eventos da UI.
- Validar entradas básicas da interface antes de disparar o pipeline.
- Atualizar o resumo de configuração em tempo real.
- Atualizar estado do botão e do badge de status.
- Usar `threading` ou `concurrent.futures` se o pipeline for executado em background.

**Não fazer**
- Não deve conter lógica de extração espectral, treino, avaliação ou classificação.
- Não deve manipular diretamente raster ou shapefile além de ler caminhos.

### 3.3 Dados e configuração de pipeline

**Fazer**
- Criar um conversor `get_pipeline_config()` no controlador que gera o objeto de configuração completo.
- Validar que arquivos existem antes de iniciar.
- Garantir que campos obrigatórios estejam preenchidos (imagem treino, imagem classificação, saída, shapefiles, modelo, etc.).
- Tratar ações de modelo:
  - `Treinar modelo novo`
  - `Treinar modelo existente`
  - `Usar modelo existente`

**Não fazer**
- Não construa caminhos hardcoded na UI.
- Não permita iniciar sem validação de entrada.

## 4. Passo a passo de implementação

### Etapa 1 — Modularizar a configuração e validação

1. Definir `PipelineConfig` no novo módulo `core/pipeline_config.py`.
2. Mapear cada campo da UI para atributos do objeto.
3. Validar: caminhos existem, imagens são GeoTIFF, shapefiles existem, modelo existente é válido.
4. Expor mensagens para a UI em caso de erro.

Métrica de sucesso
- `PipelineConfig` aceita dados da UI e retorna objeto sem exceção para entradas válidas.
- Erros de validação são retornados com mensagens claras.
- Testes unitários cobrem casos válidos e inválidos.

### Etapa 2 — Criar componentes de leitura de dados e hardware

1. Criar `HardwareManager` (info TF, RAM, CPU, chunk size).
2. Criar `RasterSource` para treino / classificação.
3. Criar `ShapefileDataset` para carregar shapefiles e unir CRS.
4. Atualizar `MainController` para obter e exibir `HardwareManager` no log/resumo.

Métrica de sucesso
- `HardwareManager` retorna `device`, `ram_total_gb`, `ram_limite_gb`, `cpu_threads`.
- `RasterSource` abre raster sem vazar handlers.
- `ShapefileDataset` retorna `GeoDataFrame` com CRS compatível.

### Etapa 3 — Extrair e preparar dados

1. Implementar `FeatureExtractor` para gerar `X, Y` a partir de shapefile+imagem.
2. Implementar `DatasetSplitter` com `train_test_split(stratify=Y)`.
3. Incluir contagem de classes e shape de entrada.

Métrica de sucesso
- `X.shape[1]` corresponde ao número correto de bandas de feature.
- Classes têm distribuição estratificada e sem vazios no conjunto de teste.
- Relatório de contagens por classe é produzido.

### Etapa 4 — Construir e treinar modelo

1. Implementar `ModelFactory` com camadas ocultas, ativação e dropout.
2. Implementar `Trainer` para treinamento e salvamento opcional.
3. Tratar modelos binários vs multiclasse corretamente.

Métrica de sucesso
- Modelo compila sem erro com os parâmetros da UI.
- `Trainer` salva modelo quando `save_model=True`.
- Histórico de treino é retornado com métricas `loss` e `accuracy`.

### Etapa 5 — Avaliar e gerar relatórios

1. Implementar `Evaluator` com `classification_report`, matriz de confusão e gráficos.
2. Salvar relatório em `resultado/` e retornar métricas chave.
3. Permitir exibir resumo de métricas na UI.

Métrica de sucesso
- Avaliação fornece `accuracy`, `precision`, `recall`, `f1` por classe.
- Matriz de confusão e gráficos são gerados corretamente.
- Erros de avaliação são tratados e exibidos no log.

### Etapa 6 — Predição e exportação GeoTIFF

1. Implementar `RasterPredictor` para chunked prediction.
2. Usar `ProcessPoolExecutor` / batch pred dependendo da ordem de execução.
3. Gravar GeoTIFF com `nodata`, `compress`, `tiled` e `overviews`.
4. Atualizar UI com progresso e estimativa.

Métrica de sucesso
- Saída GeoTIFF abre em GIS e possui `nodata=255`.
- Overviews internos são criados.
- Tempo de predição/report de desempenho é consistente com chunk size.

### Etapa 7 — Orquestração completa na UI

1. Criar `ClassifierPipeline` que recebe `PipelineConfig` e executa todas as etapas.
2. `MainController` injeta `ClassifierPipeline` e chama `execute()` em background.
3. UI deve bloquear ações enquanto o pipeline estiver rodando e mostrar progresso.
4. Exibir falhas e resultados no `txt_log` e `badge_status`.

Métrica de sucesso
- Pipeline roda de ponta a ponta pela UI sem travar a interface.
- Status muda para `EXECUTANDO`, `CONCLUIDO` e `ERRO` conforme necessário.
- Logs da UI mostram etapas completas e tempo de execução.

## 5. O que fazer em cada etapa

### 5.1 Configuração e validação

- Fazer: validar caminhos e tipos de arquivo.
- Não fazer: deixar a UI iniciar execução com configurações incompletas.
- Validar: shapefile deve ter `.shp`, `.dbf`, `.shx`; modelo `.keras` se necessário.

### 5.2 Carregamento de dados

- Fazer: usar classes de wrapper para abrir arquivos.
- Não fazer: manter raster/shapefile em operações diretas dentro do controlador.
- Validar: CRS do shapefile deve ser convertido para CRS do raster.

### 5.3 Preparação de treino/teste

- Fazer: extrair valores a partir de coordenadas pontuais.
- Não fazer: assumir que a última banda é sempre máscara sem configuração.
- Validar: `n_bandas_feature` deve coincidir entre treino e classificação.

### 5.4 Modelo e treino

- Fazer: usar funções claras para arquitetura, perda e saída.
- Não fazer: construir o modelo diretamente no controlador.
- Validar: modelo compila e treina; métricas de validação melhoram ou não divergem.

### 5.5 Avaliação

- Fazer: gerar relatório por classe e matriz de confusão.
- Não fazer: depender apenas de acurácia global.
- Validar: `classification_report` deve mostrar valores numéricos e não `nan`.

### 5.6 Predição em batch

- Fazer: dividir a imagem em chunks e processar cada chunk.
- Não fazer: carregar toda imagem em memória se for grande.
- Validar: output tem tamanho correto e respeita máscara / nodata.

### 5.7 Integração UI

- Fazer: executar pipeline como tarefa de fundo.
- Não fazer: bloquear o loop principal da interface.
- Validar: UI continua responsiva e progress bar avança.

## 6. Métricas de validação de sucesso

- `PipelineConfig` validado: 100% dos campos obrigatórios preenchidos.
- `HardwareManager`: `ram_limite_gb` calculado corretamente e `device` detectado.
- `ShapefileDataset`: CRS convertido corretamente para CRS do raster.
- `FeatureExtractor`: `X` contém o número correto de características e não valor `NaN`.
- `DatasetSplitter`: classes distribuídas proporcionalmente em treino/teste (`stratify=True`).
- `Trainer`: modelo treinado com pelo menos uma época sem falha.
- `Evaluator`: relatório completo sem `nan` ou `inf`; matriz de confusão gerada.
- `RasterPredictor`: GeoTIFF de saída válido com `nodata=255` e overviews criados.
- `MainController`: UI atualiza resumo, log e badge de status corretamente.

## 7. O que não fazer

- Não introduzir lógica pesada no `ui_main.py`.
- Não deixar o controlador manipular diretamente a inferência ou salvamento de arquivos.
- Não rodar o pipeline na thread principal da GUI.
- Não usar caminhos hardcoded fora dos componentes de configuração.
- Não assumir que input de usuário é sempre válido.

## 8. Arquivos e módulos recomendados

- `core/pipeline_config.py`
- `core/hardware_manager.py`
- `core/raster_source.py`
- `core/shapefile_dataset.py`
- `core/feature_extractor.py`
- `core/dataset_splitter.py`
- `core/model_factory.py`
- `core/trainer.py`
- `core/evaluator.py`
- `core/raster_predictor.py`
- `core/classifier_pipeline.py`

## 9. Próximo passo imediato

1. Criar o primeiro módulo `PipelineConfig` para receber valores da UI.
2. Atualizar `MainController.get_pipeline_config()` para retornar toda a configuração.
3. Implementar a validação de entrada e testar manualmente na UI.
4. Inserir logs de inicialização no `txt_log`.

---

*Esse plano está preparado para ser aplicado diretamente na UI atual (`ui_main.py` + `core/main_controller.py`) com integração eventual ao código de pipeline de `main6_multcore.py`.*
