# Koncepcja programu "Audio Radar" do wizualizacji kierunku dźwięku

## Wprowadzenie

Celem programu jest stworzenie narzędzia, które w czasie rzeczywistym przechwytuje dźwięk generowany przez grę i tłumaczy go na czytelny sygnał wizualny wskazujący kierunek źródła kroków oraz strzałów przeciwnika. System ma działać na systemie Windows 11, korzystać z języka Python oraz minimalizować opóźnienie pomiędzy odebraniem dźwięku a prezentacją informacji na ekranie.

## Ogólny przepływ danych

1. **Loopback audio** – rejestrowanie dźwięku gry poprzez WASAPI loopback (np. PyAudioWPatch).
2. **Przetwarzanie sygnału** – filtrowanie, kompresja i detekcja transjentów pozwalające na wyodrębnienie kroków i strzałów.
3. **Analiza kierunku** – porównanie energii w kanałach stereo/5.1/7.1 w momencie zdarzenia.
4. **Wizualizacja** – prezentacja wyników jako radar 2D z ikonami zdarzeń.

## Przechwytywanie audio (loopback)

- Użycie bibliotek `PyAudio` lub `sounddevice` z obsługą WASAPI loopback (np. `PyAudioWPatch`).
- Otwarcie strumienia wejściowego z urządzenia odpowiadającego głośnikom/słuchawkom (Stereo Mix/What U Hear).
- Parametry: 16-bit PCM, 44.1 kHz lub 48 kHz, wielokanałowość zgodna z konfiguracją systemu (stereo/5.1/7.1).
- Buforowanie blokami 20–50 ms; przetwarzanie w osobnym wątku asynchronicznym dla niskiego opóźnienia.

## Wykrywanie zdarzeń dźwiękowych

### Filtracja i przygotowanie sygnału

- **Filtry pasmowe**: jeden tor skupiony na pasmach 2–8 kHz (kroki), drugi o szerszym paśmie dla strzałów.
- **Redukcja szumu**: np. RNNoise lub prosta bramka szumów do tłumienia ciągłego tła.
- **Kompresja dynamiki**: moduł DRC wyrównujący poziomy sygnału, aby ciche kroki nie ginęły w głośnych wybuchach.

### Detekcja kroków

- Analiza energii w oknach 10–20 ms.
- Wyszukiwanie sekwencji impulsów o powtarzalności 1,5–3 Hz (tempo chodzenia/biegu).
- Odróżnianie pojedynczych kroków od szumu poprzez minimalny próg energii i wymaganie serii co najmniej dwóch impulsów.

### Detekcja strzałów

- Wykrywanie pojedynczych impulsów o bardzo dużej amplitudzie i krótkim czasie trwania (<150 ms).
- Weryfikacja szerokiego spektrum (energia równocześnie w pasmach niskich i wysokich).
- Możliwość zastosowania klasyfikatora ML (np. SVM/MFCC) dla lepszego rozróżniania strzałów od innych impulsów.

### Klasyfikacja i walidacja zdarzeń

- Ekstrakcja cech (widmo FFT, MFCC, długość trwania, wskaźniki energii w pasmach).
- Proste reguły progowe lub lekki model ML do odróżnienia krok/strzał.
- Odrzucanie zdarzeń niepasujących do wzorców (muzyka, dialogi, pojazdy).

## Szacowanie kierunku źródła

### Tryb stereo

- Porównanie energii lewego i prawego kanału.
- Klasyfikacja do sektorów: lewy, lewy przód, przód, prawy przód, prawy.
- Możliwe heurystyki HRTF (np. różnice widma) dla przybliżonego rozróżnienia przód/tył.

### Tryb 5.1 / 7.1

- Mapowanie kanałów na kąty (Front Left ~30°, Rear Left ~120°, itp.).
- Obliczanie wektora kierunku przez ważone średnie energii w kanałach.
- Ignorowanie kanału LFE; interpolacja między kanałami gdy energia rozkłada się na kilka kanałów.
- Obsługa wirtualizacji (Windows Sonic/Dolby Atmos) w razie braku natywnego surround – choć preferowane są realne kanały.

### Dodatkowe wskazówki

- Zachowanie historii zdarzeń dla wygładzenia kierunku (np. średnia ruchoma).
- Osobne śledzenie kroków i strzałów dla możliwości niezależnej wizualizacji.

## Wizualizacja radarowa

- Okrąg 2D reprezentujący 360° wokół gracza, aktualizowany w 60 FPS.
- Kroki oznaczane np. kolorem zielonym/niebieskim (kółko), strzały czerwonym/pomarańczowym (gwiazdka).
- Intensywność/rozmiar znacznika proporcjonalne do amplitudy (proxy odległości).
- Możliwość ustawienia okna jako półprzezroczystej nakładki „always-on-top”.
- Interfejs w Pythonie: Pygame, PyQt lub Tkinter z Canvasem.

## Architektura oprogramowania

```text
AudioCaptureThread
    ↳ AudioBuffer (kolejka próbek)
        ↳ EventDetector
            ↳ StepClassifier / ShotClassifier
                ↳ DirectionEstimator
                    ↳ VisualizationController
```

- **AudioCaptureThread** – odbiera próbki z WASAPI i umieszcza w buforze cyklicznym.
- **EventDetector** – wykonuje filtrację, kompresję i detekcję transjentów.
- **Classifiers** – rozróżniają typ zdarzenia i przekazują dane o amplitudzie.
- **DirectionEstimator** – przetwarza wielokanałowy wektor energii na kąt.
- **VisualizationController** – zarządza widokiem radarowym, animacją i wygaszaniem starych zdarzeń.

## Wydajność i opóźnienie

- Cel: całkowite opóźnienie <80 ms (loopback + analiza + renderowanie).
- Utrzymywanie małych bloków audio i współdzielenie danych przez kolejki bez kopiowania.
- Wykorzystanie bibliotek numerycznych (`numpy`, opcjonalnie `numba`) dla szybkiego FFT.
- Profilowanie i opcjonalna optymalizacja w Cythonie, jeśli Python okaże się wąskim gardłem.

## Możliwości rozbudowy

- Integracja z bardziej zaawansowanym modelem ML (np. CNN na spektrogramach).
- Adaptacyjne progi czułości w zależności od szumu tła gry.
- Zapisywanie zdarzeń do logów/telemetrii (analiza po meczu).
- API pluginów umożliwiające łatwe dodanie nowych typów zdarzeń (np. przeładowanie broni).

## Podsumowanie

Prototyp "Audio Radar" korzysta z loopbacku audio i lekkich algorytmów DSP, aby w czasie rzeczywistym identyfikować kroki i strzały, a następnie tłumaczyć je na wskaźniki kierunku. Największym wyzwaniem pozostaje precyzyjne odfiltrowanie szumu oraz dostrojenie progów, jednak przy wykorzystaniu wielokanałowego dźwięku 7.1 można uzyskać znacznie większą dokładność niż w trybie stereo. System stanowi wsparcie dla graczy potrzebujących dodatkowej informacji przestrzennej, w tym osób niedosłyszących.
