import ctypes
import math
import queue
import threading
import time

import customtkinter as ctk
import cv2
import mss
import numpy as np
import pydirectinput
from PIL import Image, ImageDraw

try:
    import keyboard
except ImportError:
    keyboard = None

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

try:  # timer do Windows em 1 ms (senão time.sleep(0.01) dorme ~15 ms)
    ctypes.windll.winmm.timeBeginPeriod(1)
except Exception:
    pass

TITULO = "PB Macro - by DrimGPO"
AZUL = "#3b8ed0"
VERDE, VERDE_H = "#27ae60", "#2ecc71"
VERMELHO, VERMELHO_H = "#c0392b", "#e74c3c"
ESCALA = 0.5  # fator de redução da imagem para a visão computacional

ATRASO_INPUT = 0.03   # s: tempo do jogo para reagir ao keyUp (compensado na previsão)
IGNORAR_FANTASMA = 0.9  # s: depois de soltar, ignora o anel-fantasma no mesmo lugar
JANELA_CINZA = 2.5      # s: a bola cinza só vale até este tempo depois da última bola verde

pydirectinput.PAUSE = 0
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# faixas HSV (OpenCV: H 0-180, S/V 0-255)
# anel: verde vivo e claro. O gramado do mapa também é verde, mas mais escuro/menos vivo
# (V~95-120, S~95-125); o anel tem S>=140 e V>=175, então o corte abaixo separa os dois.
ANEL_VERDE_LO, ANEL_VERDE_HI = np.array([48, 130, 150]), np.array([74, 255, 255])
# disco que enche: verde-claro (H~66, S~90, V~235). O fundo/gramado atrás do anel vazio fica
# em V~100-145, então V>=200 evita contar o cenário como "disco".
DISCO_LO, DISCO_HI = np.array([58, 60, 200]), np.array([74, 255, 255])


class Parar(Exception):
    pass


# ---------- visão ----------
class Tela:
    """Captura da tela. Cada thread deve criar a sua própria instância."""

    def __init__(self):
        self.sct = mss.mss()
        self.box = self.sct.monitors[1]  # monitor principal

    def grab(self):
        """Frame completo em BGRA (sem cópia)."""
        shot = self.sct.grab(self.box)
        return np.frombuffer(shot.bgra, np.uint8).reshape(shot.height, shot.width, 4)

    def pequena(self):
        """Frame já reduzido (ESCALA) em RGB, pronto para a detecção."""
        return reduzir(self.grab())

    def fechar(self):
        try:
            self.sct.close()
        except Exception:
            pass


def reduzir(bgra):
    small = cv2.resize(bgra, None, fx=ESCALA, fy=ESCALA, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGRA2RGB)


def _preenchimento(hsv, cx, cy, raio):
    """Quanto do anel já foi preenchido pelo disco verde-claro (0.0 a 1.0)."""
    h_s, w_s = hsv.shape[:2]
    r_ref = 0.86 * raio
    x0, x1 = max(0, int(cx - r_ref)), min(w_s, int(cx + r_ref) + 1)
    y0, y1 = max(0, int(cy - r_ref)), min(h_s, int(cy + r_ref) + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    yy, xx = np.ogrid[y0:y1, x0:x1]
    dentro = (xx - cx) ** 2 + (yy - cy) ** 2 <= r_ref ** 2
    pixels = cv2.inRange(hsv[y0:y1, x0:x1], DISCO_LO, DISCO_HI).astype(bool) & dentro
    return min(1.0, math.sqrt(pixels.sum() / math.pi) / r_ref)


def _cobertura(mask, cx, cy, r):
    """Fração dos 48 raios que encontram verde de anel numa faixa fina ao redor de r.

    Serve para confirmar que existe mesmo um anel redondo ali, mesmo que parte dele esteja
    escondida por texto/personagem (aí a cobertura cai, mas continua alta para um anel real).
    """
    h_s, w_s = mask.shape[:2]
    rhos = np.arange(0.88 * r, 1.14 * r, 1.0)
    achou = 0
    angulos = np.linspace(0, 2 * math.pi, 48, endpoint=False)
    for ang in angulos:
        xs = (cx + rhos * math.cos(ang)).astype(int)
        ys = (cy + rhos * math.sin(ang)).astype(int)
        ok = (xs >= 0) & (xs < w_s) & (ys >= 0) & (ys < h_s)
        if ok.any() and mask[ys[ok], xs[ok]].any():
            achou += 1
    return achou / len(angulos)


def _montar_bola(hsv, mask_verde, cx, cy, raio):
    # tem que ser um anel (o miolo não pode ser verde saturado sólido: é o "flash" do Perfeito)
    r = int(raio * 0.36)
    miolo = mask_verde[max(0, int(cy) - r):int(cy) + r + 1, max(0, int(cx) - r):int(cx) + r + 1]
    if miolo.size > 0 and (cv2.countNonZero(miolo) / miolo.size) > 0.55:
        return None
    return {
        "box": (cx - raio, cy - raio, 2 * raio, 2 * raio),
        "centro": (cx, cy),
        "raio": raio,
        "fill": _preenchimento(hsv, cx, cy, raio),
    }


def achar_bolas_verdes(small):
    """Bolas verdes (anel verde de tamanho fixo + disco interno que enche).

    Retorna lista (maior primeiro) de dicts com centro/raio em pixels da imagem
    reduzida e 'fill' = fração do anel preenchida pelo disco (0..1).

    Dois passos:
      1) contornos: rápido, serve quando o anel aparece inteiro;
      2) Hough no mask verde: serve quando o anel está parcialmente escondido (texto
         "PESCA AUTOMATICA", personagem, ícone "E"), caso em que o contorno vira só
         um pedaço do anel e o centro/raio saem errados.
    """
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    mask_verde = cv2.inRange(hsv, ANEL_VERDE_LO, ANEL_VERDE_HI)
    mask_unida = cv2.morphologyEx(mask_verde, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contornos, _ = cv2.findContours(mask_unida, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h_s, w_s = small.shape[:2]
    min_d = max(18, int(w_s * 0.025))
    max_d = int(w_s * 0.25)
    bolas = []
    sobrou_pedaco = False

    for cnt in contornos:
        if cv2.contourArea(cnt) < 25:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        redondo = min_d <= min(w, h) <= max_d and 0.85 < w / h < 1.18
        if (y + h) > h_s * 0.95 or not redondo:
            sobrou_pedaco = True   # pedaço de anel (ou ruído): o Hough confere depois
            continue
        cx, cy = x + w / 2, y + h / 2
        raio = max(w, h) / 2
        if _cobertura(mask_verde, cx, cy, raio) < 0.70:
            sobrou_pedaco = True
            continue
        b = _montar_bola(hsv, mask_verde, cx, cy, raio)
        if b:
            bolas.append(b)

    if sobrou_pedaco:
        grande = cv2.resize(cv2.GaussianBlur(mask_verde, (5, 5), 1.0), None, fx=0.5, fy=0.5,
                            interpolation=cv2.INTER_AREA)
        circulos = cv2.HoughCircles(grande, cv2.HOUGH_GRADIENT, dp=1, minDist=max(8, min_d // 3),
                                    param1=60, param2=14,
                                    minRadius=max(4, min_d // 4), maxRadius=max_d // 4 + 2)
        if circulos is not None:
            for cx, cy, r in circulos[0][:20]:
                cx, cy, r = cx * 2.0, cy * 2.0, r * 2.0
                if (cy + r) > h_s * 0.95 or not (min_d / 2 <= r <= max_d / 2):
                    continue
                if any(math.hypot(cx - b["centro"][0], cy - b["centro"][1])
                       < 0.6 * max(r, b["raio"]) for b in bolas):
                    continue
                # o raio do Hough pega a linha do meio da faixa fina: procura o raio externo
                melhor = max(((_cobertura(mask_verde, cx, cy, rr), rr)
                              for rr in (r * 0.95, r, r * 1.05, r * 1.1)), key=lambda t: t[0])
                if melhor[0] < 0.60:
                    continue
                b = _montar_bola(hsv, mask_verde, cx, cy, melhor[1])
                if b:
                    bolas.append(b)

    bolas.sort(key=lambda b: -b["raio"])
    return bolas


def _pontuar_anel_cinza(hsv, cx, cy, r):
    """Fração de raios (36) que cruzam um aro claro COM contorno escuro.

    Se o aro for verde saturado (anel de bola verde), devolve 0: não é a bola cinza.
    O verde que sobe por dentro da bola cinza fica longe da maioria dos raios,
    então não atrapalha.
    """
    h_s, w_s = hsv.shape[:2]
    rhos = np.arange(0.78 * r, 1.25 * r, 1.0)
    ok = verdes = 0
    for ang in np.linspace(0, 2 * math.pi, 36, endpoint=False):
        xs = np.clip((cx + rhos * math.cos(ang)).astype(int), 0, w_s - 1)
        ys = np.clip((cy + rhos * math.sin(ang)).astype(int), 0, h_s - 1)
        h, s, v = hsv[ys, xs, 0], hsv[ys, xs, 1], hsv[ys, xs, 2]
        verde = (h >= 48) & (h <= 74) & (s >= 130) & (v >= 150)
        if verde.sum() >= 3:
            verdes += 1
        if ((v >= 180) & (s <= 95)).any() and (v <= 95).any():
            ok += 1
    if verdes / 36.0 > 0.45:
        return 0.0
    return ok / 36.0


def achar_bola_cinza(small, minimo=0.85):
    """Bola cinza ("TOQUE"): anel claro de contorno escuro, centralizado acima do personagem."""
    h_s, w_s = small.shape[:2]
    x0, x1 = int(w_s * 0.34), int(w_s * 0.66)
    y0, y1 = int(h_s * 0.18), int(h_s * 0.68)
    roi = small[y0:y1, x0:x1]

    gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY), (5, 5), 1.2)
    min_r, max_r = int(w_s * 0.03), int(w_s * 0.09)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(10, min_r),
                               param1=100, param2=22, minRadius=min_r, maxRadius=max_r)
    if circles is None:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    melhor = None
    for cx, cy, r in circles[0][:15]:
        pontos = _pontuar_anel_cinza(hsv, float(cx), float(cy), float(r))
        if pontos >= minimo and (melhor is None or pontos > melhor["pontos"]):
            melhor = {"centro": (float(cx) + x0, float(cy) + y0), "raio": float(r),
                      "pontos": pontos}
    return melhor


# ---------- app ----------
class App:
    def __init__(self, root):
        self.root = root
        root.title(TITULO)
        root.geometry("360x720")
        root.resizable(False, False)
        root.attributes("-topmost", True)

        self.cfg = {}
        self.parar = threading.Event()
        self.thread = None
        self.fila = queue.Queue()

        # cabeçalho
        ctk.CTkLabel(root, text=TITULO, font=ctk.CTkFont(size=28, weight="bold"))\
            .pack(pady=(16, 6))
        self.lbl_loop = ctk.CTkLabel(root, text="Macro: OFF", text_color=AZUL)
        self.lbl_loop.pack()
        self.lbl_status = ctk.CTkLabel(root, text="Status: Parado", text_color="gray70")
        self.lbl_status.pack(pady=(4, 10))

        # abas
        abas = ctk.CTkTabview(root, width=320, height=375)
        abas.pack(padx=20)
        aba_macro = abas.add("Macro")
        aba_det = abas.add("Detecção")

        self.v_vara = self.slider(aba_macro, "Segurar a vara", 0.5, 6, 3.0, "{}: {:.1f} s", 55)
        self.v_ms = self.slider(aba_macro, "Intervalo toques (bola cinza)", 15, 150, 40,
                                "{}: {:.0f} ms", 45)
        self.sw_topo = ctk.CTkSwitch(aba_macro, text="Sempre no topo", command=self.alternar_topo)
        self.sw_topo.select()
        self.sw_topo.pack(pady=(18, 6))
        self.sw_esconder = ctk.CTkSwitch(aba_macro, text="Minimizar ao iniciar")
        self.sw_esconder.select()
        self.sw_esconder.pack(pady=6)

        self.v_soltar = self.slider(aba_det, "Soltar E (já com previsão)", 85, 100, 98,
                                    "{}: {:.0f}%", 15)
        ctk.CTkButton(aba_det, text="Testar detecção na tela (F8)", fg_color="gray30",
                      hover_color="gray40", command=self.testar).pack(pady=10, padx=10, fill="x")
        self.txt = ctk.CTkTextbox(aba_det, height=170, state="disabled")
        self.txt.pack(padx=6, fill="x")

        # botão principal
        self.btn = ctk.CTkButton(root, text="Iniciar (F6)", height=46, fg_color=VERDE,
                                 hover_color=VERDE_H, font=ctk.CTkFont(size=15, weight="bold"),
                                 command=self.alternar)
        self.btn.pack(fill="x", padx=20, pady=14)
        ctk.CTkLabel(root, text="F6 liga/desliga  •  F8 salva foto da detecção",
                     text_color="gray60", font=ctk.CTkFont(size=11)).pack()

        if keyboard:
            # os atalhos rodam em outra thread: só enfileiram, quem executa é o Tk
            keyboard.add_hotkey("f6", lambda: self.fila.put(("cmd", "alternar")))
            keyboard.add_hotkey("f8", lambda: self.fila.put(("cmd", "testar")))
        self.drenar_fila()

    # --- widgets ---
    def slider(self, pai, texto, minimo, maximo, inicial, formato, passos):
        lbl = ctk.CTkLabel(pai, text=formato.format(texto, inicial))
        lbl.pack(pady=(14, 2))
        var = ctk.DoubleVar(value=inicial)
        ctk.CTkSlider(pai, from_=minimo, to=maximo, number_of_steps=passos, variable=var,
                      command=lambda v: lbl.configure(text=formato.format(texto, float(v))))\
            .pack(fill="x", padx=14)
        return var

    def alternar_topo(self):
        self.root.attributes("-topmost", bool(self.sw_topo.get()))

    # --- mensagens vindas da thread ---
    def log(self, msg):
        self.fila.put(("log", msg))

    def status(self, msg):
        self.fila.put(("status", msg))

    def drenar_fila(self):
        while not self.fila.empty():
            tipo, msg = self.fila.get()
            if tipo == "log":
                self.txt.configure(state="normal")
                self.txt.insert("end", msg + "\n")
                self.txt.see("end")
                self.txt.configure(state="disabled")
            elif tipo == "status":
                self.lbl_status.configure(text=f"Status: {msg}")
            elif tipo == "parado":
                self.marcar_parado()
            elif tipo == "cmd":
                getattr(self, msg)()
        self.root.after(100, self.drenar_fila)

    # --- teste ---
    def testar(self):
        tela = Tela()
        try:
            bgra = tela.grab()
            small = reduzir(bgra)
            img = Image.fromarray(cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB))
        finally:
            tela.fechar()

        verdes = achar_bolas_verdes(small)
        cinza = achar_bola_cinza(small)
        d = ImageDraw.Draw(img)
        for b in verdes:
            x, y, w, h = (v / ESCALA for v in b["box"])
            d.rectangle([x, y, x + w, y + h], outline=(0, 255, 0), width=3)
            d.text((x, max(0, y - 14)), f"enchido {b['fill'] * 100:.0f}%", fill=(0, 255, 0))
        if cinza:
            cx, cy = (v / ESCALA for v in cinza["centro"])
            r = cinza["raio"] / ESCALA
            d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(0, 180, 255), width=4)
        img.save("debug_tela.png")

        resumo = ", ".join(f"{b['fill'] * 100:.0f}%" for b in verdes) or "-"
        self.log(f"Verdes: {len(verdes)} (enchido: {resumo}) | "
                 f"Cinza: {'SIM' if cinza else 'NÃO'}")

    # --- controle ---
    def alternar(self):
        if self.thread and self.thread.is_alive():
            self.parar.set()
            return
        self.cfg = {
            "vara": self.v_vara.get(),
            "ms": self.v_ms.get() / 1000,
            "soltar": self.v_soltar.get() / 100,
        }
        self.parar.clear()
        self.thread = threading.Thread(target=self.rodar, daemon=True)
        self.thread.start()
        self.lbl_loop.configure(text="Macro: ON", text_color=VERDE_H)
        self.btn.configure(text="Parar (F6)", fg_color=VERMELHO, hover_color=VERMELHO_H)
        # a janela cobre o canto esquerdo da tela (onde as bolas podem aparecer)
        if keyboard and self.sw_esconder.get():
            self.root.iconify()

    def marcar_parado(self):
        self.lbl_loop.configure(text="Macro: OFF", text_color=AZUL)
        self.lbl_status.configure(text="Status: Parado")
        self.btn.configure(text="Iniciar (F6)", fg_color=VERDE, hover_color=VERDE_H)
        self.root.deiconify()

    # --- worker ---
    def checar(self):
        if self.parar.is_set():
            raise Parar

    def esperar(self, s):
        fim = time.time() + s
        while time.time() < fim:
            self.checar()
            time.sleep(0.02)

    def tocar_e(self):
        """Um toque rápido no E."""
        pydirectinput.keyDown("e")
        time.sleep(0.012)
        pydirectinput.keyUp("e")
        time.sleep(self.cfg["ms"])

    def rodar(self):
        tela = None
        try:
            tela = Tela()
            self.status("começando em 5s (clique no Roblox)")
            self.esperar(5)
            while True:
                self.ciclo(tela)
        except Parar:
            pass
        except Exception as e:
            self.log(f"Erro: {e}")
        finally:
            try:
                pydirectinput.keyUp("e")
                pydirectinput.mouseUp()
            finally:
                if tela:
                    tela.fechar()
                self.fila.put(("parado", None))

    def ciclo(self, tela):
        self.status("lançando a vara")
        pydirectinput.mouseDown()
        try:
            self.esperar(self.cfg["vara"])
        finally:
            pydirectinput.mouseUp()

        # espera a boia cair na água antes de começar a detectar
        self.esperar(0.6)

        self.status("aguardando / bolas verdes")
        if self.minigame(tela):
            self.log("Peixe pescado! Recomeçando...")
            self.esperar(1.5)
        else:
            self.log("Tempo esgotado; lançando novamente.")

    def fase_cinza(self, tela):
        """Bola cinza: aperta o E exatamente 10 vezes por rodada."""
        self.status("bola cinza -> 10 toques!")
        for rodada in range(1, 4):
            self.log(f"Bola cinza: 10 toques no E (rodada {rodada})")
            for _ in range(10):
                self.checar()
                self.tocar_e()
            self.esperar(0.25)  # dá tempo da bola sumir se completou
            if achar_bola_cinza(tela.pequena()) is None:
                return
        self.log("Bola cinza ainda na tela depois de 3 rodadas.")

    @staticmethod
    def _velocidade(hist):
        """Velocidade (fração/s) com que o disco enche, pelas últimas amostras."""
        if len(hist) < 3:
            return 0.0
        dt = hist[-1][0] - hist[0][0]
        if dt <= 0:
            return 0.0
        v = (hist[-1][1] - hist[0][1]) / dt
        return min(max(v, 0.0), 3.0)

    def minigame(self, tela):
        segurando = False
        centro_ativo = None
        raio_ativo = 0.0
        t_press = 0.0
        ignorado = None          # (cx, cy, raio, instante) da bola que acabamos de soltar
        hist = []                # (tempo, fill) da bola ativa
        n_bolas = 0
        ultima_atividade = time.time()
        fim_max = time.time() + 60
        ausencia = 0             # frames seguidos sem candidata
        t_ultima_verde = None    # última vez que uma bola verde de verdade apareceu
        cinza_ant = None         # bola cinza vista no frame anterior (confirmação)
        lat = 0.05               # atraso médio de um frame do laço (captura + detecção)
        t_ant = time.time()
        frames = 0
        t_fps = time.time()

        try:
            while time.time() < fim_max:
                self.checar()
                small = tela.pequena()
                agora = time.time()
                lat = 0.7 * lat + 0.3 * (agora - t_ant)
                t_ant = agora

                frames += 1
                if agora - t_fps >= 5:
                    self.log(f"laço: {frames / (agora - t_fps):.0f} fps")
                    frames, t_fps = 0, agora

                verdes = achar_bolas_verdes(small)

                # tira o anel-fantasma da bola que acabou de ser solta
                cands = []
                for b in verdes:
                    if (ignorado is not None and agora - ignorado[3] < IGNORAR_FANTASMA and
                            math.hypot(b["centro"][0] - ignorado[0],
                                       b["centro"][1] - ignorado[1])
                            < 0.6 * max(b["raio"], ignorado[2])):
                        continue
                    cands.append(b)

                # 1) bola cinza: só existe DEPOIS das bolas verdes (aparece ~0.2 s depois da
                #    última). Sem essa trava, círculos do cenário (céu, ícones, brilho) eram
                #    confundidos com a bola cinza e o macro lançava a vara de novo cedo demais.
                #    Também exige ver a mesma bola em 2 frames seguidos.
                if (not cands and n_bolas > 0 and t_ultima_verde is not None
                        and agora - t_ultima_verde < JANELA_CINZA):
                    cinza = achar_bola_cinza(small)
                    if cinza is not None and cinza_ant is not None and math.hypot(
                            cinza["centro"][0] - cinza_ant["centro"][0],
                            cinza["centro"][1] - cinza_ant["centro"][1]) < 0.35 * cinza["raio"]:
                        if segurando:
                            pydirectinput.keyUp("e")
                            segurando = False
                        self.log(f"Bola cinza confirmada (raio {cinza['raio']:.0f}, "
                                 f"nota {cinza['pontos']:.2f})")
                        self.fase_cinza(tela)
                        return True
                    cinza_ant = cinza
                else:
                    cinza_ant = None

                # 2) escolhe a bola: a que já estamos segurando ou a maior
                bola = None
                if cands:
                    ausencia = 0
                    t_ultima_verde = agora
                    if segurando:
                        bola = min(cands, key=lambda b: math.hypot(
                            b["centro"][0] - centro_ativo[0], b["centro"][1] - centro_ativo[1]))
                    else:
                        bola = cands[0]  # lista já vem ordenada, maior primeiro
                else:
                    ausencia += 1

                if bola is not None:
                    ultima_atividade = agora
                    cx, cy = bola["centro"]

                    # apareceu uma bola em outro lugar: solta a antiga e começa a nova
                    if segurando and math.hypot(cx - centro_ativo[0],
                                                cy - centro_ativo[1]) > 0.6 * raio_ativo:
                        pydirectinput.keyUp("e")
                        segurando = False
                        time.sleep(0.03)

                    if not segurando:
                        pydirectinput.keyDown("e")
                        segurando = True
                        t_press = agora
                        n_bolas += 1
                        hist = []
                        self.status(f"bola verde {n_bolas} (segurando E)")
                    centro_ativo, raio_ativo = (cx, cy), bola["raio"]  # acompanha a deriva

                    fill = bola["fill"]
                    hist.append((agora, fill))
                    hist = hist[-5:]
                    previsto = fill + self._velocidade(hist) * (lat + ATRASO_INPUT)

                    if (previsto >= self.cfg["soltar"] or fill >= 0.995) \
                            and agora - t_press >= 0.25:
                        pydirectinput.keyUp("e")
                        segurando = False
                        ignorado = (cx, cy, bola["raio"], agora)
                        self.log(f"Bola {n_bolas}: soltou (lido {fill * 100:.0f}%, "
                                 f"previsto {previsto * 100:.0f}%)")
                    time.sleep(0.002)
                    continue

                # 3) sem bola utilizável: solta o E se a bola já sumiu
                if segurando and ausencia >= 2:
                    pydirectinput.keyUp("e")
                    segurando = False
                    self.log(f"Bola {n_bolas}: sumiu com o E ainda preso (solto)")

                if time.time() - ultima_atividade > 18:
                    return False
                time.sleep(0.002)
        finally:
            if segurando:
                pydirectinput.keyUp("e")

        return False


if __name__ == "__main__":
    root = ctk.CTk()
    App(root)
    root.mainloop()