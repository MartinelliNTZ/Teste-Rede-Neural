# 📘 PADRÃO DE SISTEMA — GUIA GENÉRICO DE ARQUITETURA

> **Propósito deste documento:** Extrair a essência, os padrões de código, a estrutura de UI (incluindo o botão de informações ⓘ), o sistema de console/progresso, o sistema de preferências persistidas e a filosofia de um sistema PySide6 de processamento em segundo plano. Este guia é **genérico e agnóstico de domínio** — serve para que qualquer IA (ou desenvolvedor) consiga recriar um sistema no **mesmo estilo e padrão**, adaptando-o a qualquer outro contexto (processamento de dados, automação, análise, etc.), **sem ser contaminado** pelos detalhes específicos do sistema original.

---

## 1. 🧠 A IDEIA CENTRAL (Essência)

O sistema segue o padrão de **"Aplicação Desktop de Processamento em Lote com Interface de Controle"**:

- O usuário **configura** parâmetros de entrada e constantes de processamento.
- O usuário **seleciona** os itens a processar (arquivos, pastas, fontes de dados).
- O sistema executa o processamento em **uma thread separada** (QThread), mantendo a interface **responsiva**.
- O progresso é reportado em tempo real via **barra de progresso** e **console de saída**.
- O console captura **todos os `print()`** do sistema (inclusive de classes internas), centralizando o log na UI.
- O usuário pode **interromper** o processamento a qualquer momento.
- As configurações são **persistidas automaticamente** em JSON e restauradas na próxima execução.
- Um botão **ⓘ** no header exibe um diálogo "Sobre" com informações do app.
- Ao final, o sistema reporta **sucesso/erro** e libera a interface.

**Filosofia de design:**
- **Separação de responsabilidades:** UI (interface) ↔ Lógica de processamento (workers) ↔ Estilo (tema).
- **Desacoplamento via sinais Qt:** a lógica de processamento **não conhece** a UI; ela apenas emite sinais.
- **Reutilização:** o console é um widget genérico que qualquer classe pode usar via redirecionamento de stdout.
- **Feedback rico:** timestamps, emojis, cores de status, ETA, tempo decorrido/restante.

---

## 2. 🏗️ ESTRUTURA DE ARQUIVOS (Padrão de Organização)

```
projeto/
├── main.py                  # Entry point + UI principal (janela, painéis, wiring)
├── requirements.txt         # Dependências
├── README.md                # Documentação
├── core/                    # Pacote com a lógica de domínio (agnóstico de UI)
│   ├── __init__.py
│   ├── styles.py            # Tema visual (cores, stylesheet global)
│   ├── preferences.py       # Sistema de preferências persistidas (JSON)
│   ├── preferences.json     # Arquivo de preferências (gerado em runtime)
│   ├── <dominio>_manager.py # Classe(s) de processamento/lógica de negócio
│   └── ...                  # Outros módulos de domínio
└── RESULT/                  # Pasta de saída (gerada em runtime)
```

**Regra de ouro:** a pasta `core/` contém **apenas lógica pura** (sem imports de Qt). A UI fica toda em `main.py`. Isso permite testar e reutilizar a lógica independentemente da interface.

---

## 3. 🎨 TEMA VISUAL (Estilo)

### 3.1 Filosofia do Tema
- Tema **Dark Premium** (fundo escuro + cor de destaque metálica/dourada).
- Inspirado em **painéis de controle** (drones, HUD, instrumentos).
- Identidade visual reforçada por **detalhes animados** (spinner, pulso de brilho).

### 3.2 Paleta de Cores (padrão `core/styles.py`)
Definir uma classe `Colors` com constantes nomeadas:

```python
class Colors:
    # Fundos
    BACKGROUND      = "#0d0d0f"   # fundo principal
    PANEL           = "#16161a"   # fundo de painéis/grupos
    BLACK_SOFT      = "#1a1a1e"   # preto suave
    # Bordas
    BORDER          = "#2a2a30"
    BORDER_LIGHT    = "#3a3a42"
    # Destaque (cor premium)
    GOLD            = "#d4af37"   # dourado principal
    GOLD_LIGHT      = "#f0d98c"   # dourado claro
    GOLD_DIM        = "#8a6d1f"   # dourado escurecido
    # Status
    SUCCESS         = "#4caf50"   # verde
    WARNING         = "#ff9800"   # laranja
    ERROR           = "#f44336"   # vermelho
    # Texto
    TEXT            = "#e0e0e0"
    TEXT_DIM        = "#9e9e9e"
```

### 3.3 Stylesheet Global (padrão `core/styles.py`)
- Classe `Styles` com método estático `get_stylesheet()` retornando um QSS completo.
- Usar `setObjectName()` nos widgets para aplicar estilos específicos via QSS.
- Padrões de nomenclatura de objectName: `headerBar`, `titleLabel`, `subtitleLabel`, `consoleWidget`, `centralWidget`, `primaryButton`, `dangerButton`.
- Botões com classes de estilo: `primaryButton` (ação principal, destaque), `dangerButton` (ações destrutivas).
- `QDialog` estilizado no QSS global (fundo `BACKGROUND`, texto `TEXT`) para que diálogos como o "Sobre" herdem o tema.

```python
class Styles:
    @staticmethod
    def get_stylesheet() -> str:
        return f"""
        QMainWindow, QWidget#centralWidget {{
            background-color: {Colors.BACKGROUND};
            color: {Colors.TEXT};
            font-family: 'Segoe UI', sans-serif;
        }}
        QGroupBox {{
            background-color: {Colors.PANEL};
            border: 1px solid {Colors.BORDER};
            border-radius: 8px;
            margin-top: 12px;
            padding: 10px;
            font-weight: 600;
        }}
        QPushButton#primaryButton {{
            background-color: {Colors.GOLD};
            color: #000;
            border-radius: 6px;
            font-weight: 700;
        }}
        QPushButton#primaryButton:hover {{
            background-color: {Colors.GOLD_LIGHT};
        }}
        QPushButton#dangerButton {{
            background-color: transparent;
            color: {Colors.ERROR};
            border: 1px solid {Colors.ERROR};
            border-radius: 6px;
        }}
        QPushButton#dangerButton:hover {{
            background-color: {Colors.ERROR};
            color: #fff;
        }}
        QPlainTextEdit#consoleWidget {{
            background-color: {Colors.BLACK_SOFT};
            border: 1px solid {Colors.BORDER};
            border-radius: 6px;
            font-family: 'Consolas', monospace;
            color: {Colors.TEXT};
        }}
        QDialog {{
            background-color: {Colors.BACKGROUND};
            color: {Colors.TEXT};
        }}
        """
```

---

## 4. 🖥️ ESTRUTURA DA INTERFACE (UI)

### 4.1 Layout Geral
```
┌──────────────────────────────────────────────────────────────┐
│ HEADER (título + subtítulo + spinner + ⓘ info + status dot)  │
├──────────────────────────────────────────────────────────────┤
│ [Barra de Progresso]                                         │
├───────────────────────────────┬──────────────────────────────┤
│  COLUNA ESQUERDA              │  COLUNA DIREITA              │
│  (Configurações)              │  (Console)                   │
│  ┌─────────────────────────┐  │  ┌────────────────────────┐  │
│  │ Painel: Constantes      │  │  │ Console (QPlainTextEdit)│  │
│  ├─────────────────────────┤  │  │                        │  │
│  │ Painel: Seleção de      │  │  │                        │  │
│  │ arquivos (lista)        │  │  │                        │  │
│  ├─────────────────────────┤  │  ├────────────────────────┤  │
│  │ Painel: Pasta destino   │  │  │ [Limpar][Copiar][Testar]│  │
│  ├─────────────────────────┤  │  └────────────────────────┘  │
│  │ [🚀 PROCESSAR]          │  │                              │
│  └─────────────────────────┘  │                              │
├───────────────────────────────┴──────────────────────────────┤
│ STATUS BAR                                                   │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 Componentes Padrão

**HeaderBar (QFrame):**
- Altura fixa (ex.: 78px).
- Título grande + subtítulo descritivo.
- Spinner animado decorativo à esquerda.
- Botão pequeno de informações (ⓘ) à direita, antes do status, abrindo o diálogo "Sobre".
- Indicador de status à direita (`● PRONTO`, `● PROCESSANDO`, `● CONCLUÍDO`, `● ERRO`) com cores de status.
- Animação de "pulso" de brilho no título via QTimer (sem dependências externas).

**Painéis de Configuração (QGroupBox):**
- Cada grupo de configuração é uma classe própria (`ConstantsPanel`, `FilesPanel`, `TrajectoryPanel`).
- Cada painel expõe um método `get_*()` que retorna os valores configurados (ex.: `get_constants()`, `get_files()`, `get_path()`).
- Uso de `QSpinBox`/`QDoubleSpinBox` para valores numéricos com ranges e steps definidos.
- Uso de `QListWidget` com `ExtendedSelection` para listas de arquivos.
- Botões de ação com emojis no texto (`➕`, `➖`, `🗑️`, `📂`).

**Splitter:**
- `QSplitter(Qt.Horizontal)` separando configurações (esquerda) e console (direita).
- `setStretchFactor(0, 0)` e `setStretchFactor(1, 1)` — console expande, configurações mantêm tamanho.
- `setSizes([420, 760])` — tamanhos iniciais.

**Status Bar:**
- `QStatusBar` para mensagens rápidas de feedback (com timeout, ex.: 3000ms).

### 4.3 Padrão de Criação de Painéis
Cada painel segue o padrão:
1. Herdar de `QGroupBox` (ou `QFrame`).
2. Título descritivo no construtor.
3. Layout interno (QGridLayout, QVBoxLayout, QHBoxLayout).
4. Widgets de entrada com ranges/validação.
5. Método público `get_*()` para extrair valores.
6. Conexões de sinais internas (ex.: botão → método privado `_browse`, `_add_files`).

### 4.4 Botão de Informações (ⓘ) e Diálogo "Sobre"

**Constantes do app** — declaradas logo após os imports:

```python
APP_VERSAO = "1.0.0"                      # Versão da aplicação
APP_DATA_ATUALIZACAO = "08/06/2026"      # Data da última atualização
APP_TO = "Palmas - TO"                    # Cidade/UF
APP_EMPRESA = "Linhas Brasil"             # Nome da empresa
```

**Classe ofuscada do autor (`Npb`)** — esconde o nome do autor usando **XOR + hexadecimal** (o nome não aparece em texto puro no código-fonte). Para gerar um nome ofuscado para outro autor:

```python
def ofuscar(nome: str, chave: str) -> str:
    """Gera o valor hexadecimal do nome XOR chave."""
    data = nome.encode("utf-8")
    key_bytes = chave.encode("utf-8")
    return bytes(
        b ^ key_bytes[i % len(key_bytes)]
        for i, b in enumerate(data)
    ).hex()
```

Classe pronta (genérica):

```python
class Npb:
    """Classe interna (ofuscada) que revela o autor do programa."""
    _LFGT = "COLAR_AQUI_O_HEX_GERADO"    # nome XOR chave, em hexadecimal
    _LGTR = "dfjaskhdfkjsa"               # chave usada para reverter o XOR

    @staticmethod
    def _decrypt(hex_value: str, key: str) -> str:
        data = bytes.fromhex(hex_value)
        key_bytes = key.encode("utf-8")
        return bytes(
            b ^ key_bytes[i % len(key_bytes)]
            for i, b in enumerate(data)
        ).decode("utf-8")

    @classmethod
    def npbt(cls) -> str:
        """Retorna o nome real, já descriptografado."""
        return cls._decrypt(cls._LFGT, cls._LGTR)
```

**Diálogo "Sobre" (`SobreDialog`)** — `QDialog` modal com:
- Título "ℹ️ Sobre" usando `objectName = "titleLabel"` e alinhamento central.
- Nome do aplicativo usando `objectName = "subtitleLabel"`.
- Separador (`QFrame.HLine`) com cor `Colors.BORDER`.
- Grade (`QGridLayout`) de informações: Versão, Atualização, Local, Empresa, Autor — rótulo em `Colors.GOLD`, valor em `Colors.TEXT` com `setWordWrap(True)`.
- Botão `OK` com `objectName = "primaryButton"` que fecha o diálogo.
- `QDialog` estilizado no QSS global (`background-color: BACKGROUND`).

```python
class SobreDialog(QDialog):
    """Dialogo 'Sobre' com informacoes da aplicacao."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Informações")
        self.setFixedWidth(420)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 24, 24, 24)

        # Titulo
        title = QLabel("ℹ️ Sobre")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Nome do aplicativo
        app_name = QLabel("NOME_DO_SEU_APP")
        app_name.setObjectName("subtitleLabel")
        app_name.setAlignment(Qt.AlignCenter)
        layout.addWidget(app_name)

        # Separador
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {Colors.BORDER};")
        layout.addWidget(sep)

        # Grade de informacoes
        grid = QGridLayout()
        grid.setVerticalSpacing(8)
        grid.setHorizontalSpacing(12)

        linhas = [
            ("Versão", APP_VERSAO),
            ("Atualização", APP_DATA_ATUALIZACAO),
            ("Local", APP_TO),
            ("Empresa", APP_EMPRESA),
            ("Autor", Npb.npbt()),
        ]

        for i, (rotulo, valor) in enumerate(linhas):
            lbl_rotulo = QLabel(rotulo)
            lbl_rotulo.setStyleSheet(f"color: {Colors.GOLD}; font-weight: 700;")
            lbl_valor = QLabel(valor)
            lbl_valor.setStyleSheet(f"color: {Colors.TEXT};")
            lbl_valor.setWordWrap(True)
            grid.addWidget(lbl_rotulo, i, 0, Qt.AlignTop)
            grid.addWidget(lbl_valor, i, 1)

        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)

        # Botao fechar
        btn_layout = QHBoxLayout()
        btn_layout.addStretch(1)
        btn_ok = QPushButton("OK")
        btn_ok.setObjectName("primaryButton")
        btn_ok.setFixedWidth(100)
        btn_ok.clicked.connect(self.accept)
        btn_layout.addWidget(btn_ok)
        btn_layout.addStretch(1)
        layout.addLayout(btn_layout)
```

**Botão ⓘ no header** — adicionado **antes do label de status**:

```python
btn_info = QPushButton("ⓘ")
btn_info.setToolTip("Informações")
btn_info.setFixedSize(32, 32)
btn_info.setCursor(Qt.PointingHandCursor)
btn_info.setStyleSheet(
    "QPushButton {"
    f"  background-color: {Colors.PANEL};"
    f"  color: {Colors.GOLD};"
    "  border: 1px solid " + Colors.BORDER + ";"
    "  border-radius: 16px;"
    "  font-size: 16px;"
    "  font-weight: 700;"
    "  padding: 0px;"
    "}"
    "QPushButton:hover {"
    f"  border-color: {Colors.GOLD};"
    f"  background-color: {Colors.BLACK_SOFT};"
    "}"
)
btn_info.clicked.connect(self._on_info_clicked)
header_layout.addWidget(btn_info)
```

Ordem visual no header:

```
┌──────────────────────────────────────────────────────────────┐
│  Título + Subtítulo          ⓘ        ● PRONTO               │
└──────────────────────────────────────────────────────────────┘
        ↑ título_box          ↑ botão   ↑ status_label
```

**Slot que abre o diálogo** (na MainWindow):

```python
def _on_info_clicked(self):
    """Abre o dialogo de informacoes do programa."""
    dialog = SobreDialog(self)
    dialog.exec()
```

**Ordem de classes no arquivo principal:** constantes `APP_*` → classe `Npb` → `SobreDialog` → demais classes.

---

## 5. 🖨️ SISTEMA DE CONSOLE (Log Centralizado)

### 5.1 Conceito
O console é um **widget reutilizável** que captura **toda a saída `print()`** do sistema — inclusive de classes internas que não conhecem a UI — e a exibe em tempo real.

### 5.2 Componentes

**`StreamRedirector(QObject)`** — objeto "file-like" que substitui `sys.stdout`/`sys.stderr`:
```python
class StreamRedirector(QObject):
    text_written = Signal(str)
    def write(self, text: str):
        if text:
            payload = str(text)
            if not payload.endswith("\n"):
                payload += "\n"
            self.text_written.emit(payload)
    def flush(self):
        pass
```

**`ConsoleBridge(QObject)`** — canal seguro para encaminhar mensagens da thread de trabalho para a UI (evita acesso direto a widgets de outra thread):
```python
class ConsoleBridge(QObject):
    message_written = Signal(str)
    def write(self, text: str):
        if text:
            payload = str(text)
            if not payload.endswith("\n"):
                payload += "\n"
            self.message_written.emit(payload)
    def flush(self):
        pass
```

**`ConsoleWidget(QPlainTextEdit)`** — o widget de exibição:
- `setReadOnly(True)`, `setMaximumBlockCount(5000)` (limita memória).
- `setLineWrapMode(QPlainTextEdit.NoWrap)`.
- **Buffer de linhas:** acumula texto e só materializa uma linha quando um `\n` completo chega (resolve quebra de linha incorreta, pois `print()` dispara `write()` separado para conteúdo e para `\n`).
- `appendPlainText` sempre escreve no final (resolve bug de cursor).
- Auto-scroll para o final.
- Métodos: `append_line()`, `flush_pending()`, `write_stream()`, `clear()`, `copy_all_to_clipboard()`.

### 5.3 Redirecionamento (Wiring)
```python
def _setup_stdout_redirect(self):
    self._console_bridge = ConsoleBridge()
    self._stdout_redirector = StreamRedirector()
    self._stderr_redirector = StreamRedirector()
    # QueuedConnection: atualização só na thread principal da UI
    self._console_bridge.message_written.connect(self.console.write_stream, Qt.QueuedConnection)
    self._stdout_redirector.text_written.connect(self._console_bridge.write)
    self._stderr_redirector.text_written.connect(self._console_bridge.write)
    sys.stdout = self._stdout_redirector
    sys.stderr = self._stderr_redirector
```

**Importante:** restaurar `sys.stdout = sys.__stdout__` e `sys.stderr = sys.__stderr__` no `closeEvent`.

### 5.4 Formato das Mensagens de Log
- Prefixo com timestamp: `[{HH:MM:SS}]`.
- Emojis para categorias: `✅` sucesso, `❌` erro, `⚠️` aviso, `⏳` progresso, `📂` arquivo, `⚙️` configuração, `🧭` pasta, `🧪` teste, `⏹️` interrupção.
- Caixas decorativas para eventos importantes (início/fim de processamento):
  ```
  ╔══════════════════════════════════════════╗
  ║  INICIANDO PROCESSAMENTO                  ║
  ╚══════════════════════════════════════════╝
  ```

---

## 6. 📊 BARRA DE PROGRESSO

### 6.1 Configuração
- `QProgressBar` com range `0..100`.
- Inicialmente oculta (`setVisible(False)`), aparece ao iniciar processamento.
- Formato dinâmico com informações ricas.

### 6.2 Formato Dinâmico
```
"{processed:,}/{total:,} pts · Tempo decorrido: {elapsed}  Restante: {remaining}  ETA {eta}  {percent:.3f}%"
```

### 6.3 Cálculo de ETA e Tempo Restante
```python
def _format_time_hms(self, seconds: float) -> str:
    total_seconds = int(round(max(0.0, seconds)))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

# ETA/restante
if processed > 0 and processed < total:
    remaining = elapsed * (total - processed) / processed
    remaining_text = self._format_time_hms(remaining)
    eta_dt = datetime.now() + timedelta(seconds=remaining)
    eta_text = eta_dt.strftime("%H:%M:%S")
```

### 6.4 Throttling de Log
Para não poluir o console com milhões de linhas, só logar progresso a cada N unidades ou em frações do total:
```python
if processed == total or processed - self._progress_last_update >= max(1_000_000, total // 20):
    self._progress_last_update = processed
    print(f"...")
```

---

## 7. ⚙️ PROCESSAMENTO EM SEGUNDO PLANO (QThread + Worker)

### 7.1 Padrão Worker
O worker é um `QObject` (não QThread) movido para uma QThread:

```python
class ProcessingWorker(QObject):
    started = Signal()
    finished = Signal(bool, str)
    progress = Signal(str, object, object, float)  # object evita overflow de int 32-bit
    file_completed = Signal(str, bool)

    def __init__(self, ...):
        super().__init__()
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def run(self):
        self.started.emit()
        try:
            # ... processamento ...
            self.finished.emit(True, "Concluído")
        except Exception as exc:
            self.finished.emit(False, str(exc))
```

### 7.2 Sinais do Worker
| Sinal | Assinatura | Descrição |
|-------|-----------|-----------|
| `started` | `()` | Início do processamento |
| `finished` | `(bool, str)` | Fim (sucesso, mensagem) |
| `progress` | `(str, object, object, float)` | Progresso (arquivo, processado, total, tempo) |
| `file_completed` | `(str, bool)` | Um item concluído (arquivo, sucesso) |

**⚠️ Lição importante:** usar `object` (não `int`) em sinais que carregam contagens potencialmente grandes. Sinais Qt tipados como `int` são C++ int de 32 bits (máx. ~2,1 bilhões) e disparam `OverflowError` para valores maiores. Com `object`, o int Python de precisão arbitrária trafega sem conversão.

### 7.3 Wiring do Thread
```python
self._worker = ProcessingWorker(...)
self._worker_thread = QThread(self)
self._worker.moveToThread(self._worker_thread)
self._worker_thread.started.connect(self._worker.run)
self._worker.started.connect(self._on_worker_started)
self._worker.progress.connect(self._on_worker_progress)
self._worker.file_completed.connect(self._on_worker_file_completed)
self._worker.finished.connect(self._on_worker_finished)
self._worker.finished.connect(self._worker_thread.quit)
self._worker_thread.finished.connect(self._worker_thread.deleteLater)
self._worker_thread.start()
```

### 7.4 Ciclo de Vida / Cleanup
- Ao finalizar: `self._worker_thread.quit()`, `self._worker_thread.wait()`, resetar referências.
- No `closeEvent`: `request_stop()`, `quit()`, `wait()`, restaurar stdout/stderr.
- Desabilitar botão de processar durante execução (`setEnabled(False)`), reabilitar ao final.

### 7.5 Callback de Log do Worker
O worker recebe um `log_callback` opcional para escrever no console via bridge (thread-safe):
```python
def _print(self, message: str):
    if self.log_callback is not None:
        self.log_callback(str(message) + "\n")
    else:
        print(message)
```

---

## 8. 💾 SISTEMA DE PREFERÊNCIAS (Persistência Automática)

### 8.1 Conceito
As preferências do usuário são persistidas em **JSON** dentro da pasta `core/` (`core/preferences.py` + `core/preferences.json`), de forma **agnóstica de UI**. A interface carrega os valores nos widgets ao iniciar e **salva automaticamente** a cada mudança — sem botão "salvar".

### 8.2 Estrutura do JSON
```json
{
  "compartilhadas": { "EPSG_ALVO": 4674, "EMAIL": "user@mail.com", "...": "..." },
  "ferramentas": {
    "metashape": { "ARQ_INPUT": "INPUT.txt", "...": "..." },
    "fotos": { "...": "..." }
  }
}
```

- **`compartilhadas`**: valores comuns a todas as ferramentas do sistema.
- **`ferramentas/<namespace>`**: valores específicos de cada ferramenta (isolamento por namespace).
- **Migração automática**: arquivos antigos sem a estrutura de ferramentas são tratados como compartilhadas.

### 8.3 Classe `Preferences` (padrão `core/preferences.py`)
```python
class Preferences:
    ARQUIVO_PREFERENCIAS = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "preferences.json",
    )

    DEFAULTS_COMPARTILHADAS = { ... }   # padrões globais
    DEFAULTS_FERRAMENTA_A = { ... }     # padrões da ferramenta A
    DEFAULTS_FERRAMENTA_B = { ... }     # padrões da ferramenta B

    def __init__(self, namespace: str | None = None):
        ...
        self._carregar()  # lê o JSON e mescla com defaults
        # garante que a seção da ferramenta existe com todos os defaults

    def get(self, chave, padrao=None):    # prioridade: ferramenta → compartilhada → padrao
    def set(self, chave, valor):          # salva automaticamente
    def set_muitos(self, valores: dict):  # salva vários de uma vez
    def reset(self):                      # restaura defaults e salva
    def to_dict(self) -> dict:            # cópia (compartilhadas + ferramenta)
```

Regras:
- `set()` / `set_muitos()` **salvam imediatamente** no JSON (self-saving).
- `get()` resolve na ordem: **ferramenta → compartilhada → valor padrão**.
- Arquivo corrompido ou inacessível: usa defaults silenciosamente (`try/except`).
- `blockSignals` não é usado aqui — isso é responsabilidade da UI.

### 8.4 Integração com a UI (wiring)

**Inicialização** (no `__init__` da MainWindow):
```python
self.prefs = Preferences("nome_da_ferramenta")
```

**Conexões de sinais** (no `_setup_ui`) — cada widget que representa uma preferência conecta ao slot de salvamento:
```python
self.meu_combo.currentIndexChanged.connect(self._on_pref_changed)
self.meu_edit.textChanged.connect(self._on_pref_changed)
self.meu_check.toggled.connect(self._on_pref_changed)
self.meu_spin.valueChanged.connect(self._on_pref_changed)
```

**Carregamento** — método `_carregar_preferencias()` chamado ao final do `__init__` da janela:
1. `blockSignals(True)` em todos os widgets de preferência (evita salvar valores intermediários durante o preenchimento).
2. Preenche cada widget com `self.prefs.get("CHAVE", valor_padrao)`.
3. `blockSignals(False)` no `finally`.
4. Re-salva os valores normalizados via `set_muitos()` (ex.: `.text().strip()`, `currentData()`).

```python
def _carregar_preferencias(self):
    widgets = [self.meu_combo, self.meu_edit, self.meu_check]
    for w in widgets:
        w.blockSignals(True)
    try:
        epsg = self.prefs.get("EPSG_ALVO", 4674)
        idx = self.meu_combo.findData(epsg)
        if idx >= 0:
            self.meu_combo.setCurrentIndex(idx)
        self.meu_edit.setText(self.prefs.get("MEU_CAMPO", "default"))
    finally:
        for w in widgets:
            w.blockSignals(False)
    # Valores normalizados (sem vazios indesejados)
    self.prefs.set_muitos({
        "EPSG_ALVO": self.meu_combo.currentData(),
        "MEU_CAMPO": self.meu_edit.text().strip(),
    })
```

**Salvamento automático** — método `_on_pref_changed()` conectado aos widgets:
```python
def _on_pref_changed(self):
    """Salva as preferencias quando um campo muda."""
    self.prefs.set_muitos({
        "EPSG_ALVO": self.meu_combo.currentData(),
        "MEU_CAMPO": self.meu_edit.text().strip(),
        "MEU_CHECK": self.meu_check.isChecked(),
    })
```

### 8.5 Resumo do Padrão
1. **`core/preferences.py`** contém a classe `Preferences` — lógica pura, sem Qt.
2. **Defaults** declarados como dicionários de classe (compartilhadas + por ferramenta).
3. **Namespaces** isolam preferências entre ferramentas do mesmo sistema.
4. **Self-saving**: qualquer `set()`/`set_muitos()` persiste imediatamente no JSON.
5. **UI wiring**: widgets conectam a `_on_pref_changed`; carregamento com `blockSignals`.
6. **Testável**: a lógica de preferências não depende da interface.

---

## 9. 🧩 PADRÕES DE CÓDIGO E BOAS PRÁTICAS

### 9.1 Estrutura de Classes
- **Widgets de UI** herdam de `QGroupBox`/`QFrame`/`QPlainTextEdit` e encapsulam seu próprio layout e lógica.
- **Workers** herdam de `QObject` e expõem sinais; nunca tocam a UI diretamente.
- **Lógica de domínio** fica em `core/`, sem imports de Qt.

### 9.2 Convenções de Nomenclatura
- Classes: `PascalCase` (ex.: `MainWindow`, `ConsoleWidget`, `ProcessingWorker`, `SobreDialog`).
- Métodos públicos: `snake_case` (ex.: `get_constants`, `request_stop`).
- Métodos privados: prefixo `_` (ex.: `_on_process_clicked`, `_setup_stdout_redirect`).
- Slots de sinais: prefixo `_on_` (ex.: `_on_worker_progress`, `_on_worker_finished`, `_on_pref_changed`, `_on_info_clicked`).
- objectName: camelCase (ex.: `consoleWidget`, `primaryButton`).

### 9.3 Validação de Entrada
- Antes de processar, validar entradas e reportar erro no console + status:
  ```python
  if not files:
      print(f"[{self._timestamp()}] ❌ Nenhum arquivo selecionado.")
      self.header.set_status("ERRO", Colors.ERROR)
      return
  ```

### 9.4 Tratamento de Erros
- `try/except` no worker com `traceback.format_exc()` logado no console.
- Sinais `finished(False, mensagem)` para falhas.
- Status do header atualizado para `ERRO` em caso de falha.

### 9.5 Feedback ao Usuário
- **Header status dot:** `PRONTO` / `PROCESSANDO` / `CONCLUÍDO` / `ERRO` com cores.
- **Status bar:** mensagens rápidas com timeout.
- **Console:** log detalhado com timestamps.
- **Progress bar:** percentual + tempo decorrido + restante + ETA.

---

## 10. 🚀 ENTRY POINT (main)

```python
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # estilo consistente entre plataformas
    app.setStyleSheet(Styles.get_stylesheet())  # tema global

    window = MainWindow()
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
```

---

## 11. 📋 CHECKLIST PARA RECRIAR UM SISTEMA NO MESMO PADRÃO

Ao criar um novo sistema seguindo este padrão, garanta que:

- [ ] **Estrutura:** `main.py` (UI) + `core/` (lógica pura, sem Qt) + `styles.py` (tema).
- [ ] **Tema:** Dark Premium com `Colors` (constantes) e `Styles.get_stylesheet()` (QSS global).
- [ ] **Header:** título + subtítulo + spinner animado + botão ⓘ + status dot com cores.
- [ ] **Botão info:** diálogo "Sobre" (`SobreDialog`) com constantes `APP_*`, autor ofuscado via `Npb` e botão ⓘ abrindo o diálogo.
- [ ] **Painéis de configuração:** classes `QGroupBox` com `get_*()` para extrair valores.
- [ ] **Seleção de itens:** `QListWidget` com `ExtendedSelection` + botões adicionar/remover/limpar.
- [ ] **Console:** `ConsoleWidget` com buffer de linhas, `StreamRedirector` + `ConsoleBridge`, redirecionamento de stdout/stderr, restauração no close.
- [ ] **Progress bar:** formato dinâmico com `%`, tempo decorrido, restante, ETA; throttling de log.
- [ ] **Worker:** `QObject` movido para `QThread`, sinais `started/finished/progress/file_completed`, `request_stop()`, `log_callback`.
- [ ] **Sinais de progresso:** usar `object` (não `int`) para contagens grandes.
- [ ] **Preferências:** classe `Preferences` em `core/preferences.py` (defaults + namespaces + self-saving JSON), carregamento com `blockSignals` e salvamento via `_on_pref_changed`.
- [ ] **Validação:** entradas validadas antes de processar, com feedback no console/status.
- [ ] **Erros:** `try/except` com `traceback`, `finished(False, msg)`, status `ERRO`.
- [ ] **Cleanup:** `quit()`/`wait()` do thread, restaurar stdout/stderr, reabilitar botões.
- [ ] **Logs:** timestamps `[HH:MM:SS]` + emojis por categoria + caixas decorativas para eventos importantes.

---

## 12. 🎯 RESUMO DA FILOSOFIA

> **"Interface rica e responsiva, lógica desacoplada, feedback total."**

1. **UI nunca bloqueia** — processamento sempre em thread separada.
2. **Lógica nunca conhece a UI** — comunicação apenas via sinais e callbacks.
3. **Todo output é visível** — console centralizado captura tudo, com timestamps e categorias.
4. **O usuário sempre sabe o que está acontecendo** — status, progresso, ETA, erros claros.
5. **Estilo consistente e premium** — tema dark com destaque metálico, animações sutis.
6. **Código organizado e reutilizável** — widgets encapsulados, lógica em `core/`, padrões claros.
7. **Preferências persistidas automaticamente** — configurações salvas em JSON (`core/preferences.py`) a cada mudança e restauradas na próxima execução.