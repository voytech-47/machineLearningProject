# Raport z projektu o Formule 1

06-DUMAUI0 2024/SZ

Autor: *Wojciech Grzybowski*

## Cel projektu
Celem projektu było stworzenie dwóch modeli:

1. Przewiduje nazwę zespołu na podstawie zdjęcia logo (sieci neuronowe),
2. Przewiduje pozycje końcową w klasyfikacji generalnej na podstawie następujących cech:
	1. Średnia pozycja końcowa w wyścigu w sezonie,
	2. Średnia pozycja końcowa w kwalifikacjach w sezonie,
	3. % zwycięstw w sezonie względem liczby wyścigów,
	4. % przejechanych okrążeń względem wszystkich możliwych

Dla cechy 1. należy wykorzystać stosunek zwycięstw do liczby wyścigów, ponieważ sezony mogą mieć różną liczbę wyścigów, co naturalnie implikuje tą samą zależność dla cechy 2.

## Dane
Dane dla problemu 1. pochodzą ze strony Google Grafika (logo każdego zespołu). Z pierwotnych zdjęć utworzono sztuczny zbiór, generując duplikaty oryginalnego logo, które mają nałożony losowy obrót od 0 do 360 stopni. W ten sposób utworzony został zbiór mający 2500 przykładów (250 zdjęć dla każdego z 10 zespołów)

Dane dla problemu 2. pochodzą ze strony kaggle.com: https://www.kaggle.com/datasets/cjgdev/formula-1-race-data-19502017

Aby przygotować dane do uczenia dla problemu 2. należało:
1. Odrzucić kolumny, które nie wnoszą wartościowych informacji (takich jak numer kierowcy, liczba punktów, czas wyścigu itd.),
2. Pliki *races* oraz *results* należało połączyć za pomocą kolumny *raceId*, aby określić, z którego roku pochodzi dany rezultat. W ten sposób można policzyć wszystkie statystyki w każdym z sezonów osobno, ponieważ kierowcy mogą brać udział w więcej niż jednym sezonie. Za pomocą pętli for, która inkrementuje od 1950 do 2018, wybieram tylko te wiersze, które wartość *year* mają równą ze zmienną sterującą, aby brać pod uwagę wyniki tylko z jednego sezonu.
3. Zliczenie średniej z pozycji na starcie oraz pozycji na mecie, a następnie pogrupowanie tych wyników według unikalnego id kierowcy *driverId*,
6. Aby zliczyć % zwycięstw w sezonie, należało zliczyć wszystkie wyścigi w sezonie na podstawie kolumny *year*. Następnie, dla każdego kierowcy (*driverId*) zsumować kolumnę *positionOrder* w danym sezonie, jeżeli wartość wiersza = 1 (1. miejsce na mecie - zwycięstwo w wyścigu). Taką sumę dzielę przez liczbę wyścigów i otrzymuję % zwycięstw w sezonie dla każdego kierowcy.
7. Aby zliczyć % przejechanych okrążeń w sezonie, należało zliczyć wszystkie okrążenia znajdujące się w pliku *lapTimes* dla każdego kierowcy. Liczba okrążeń składająca się na cały wyścig niestety nie jest przechowywana w pliku *races*, dlatego zakładamy, że całkowita liczba okrążeń w wyścigu to wartość funkcji max z wszystkich zarejestrowanych okrążeń dla danego wyścigu - w całej historii Formuły 1 nigdy nie zdarzyło się, aby nikt nie ukończył wyścigu, dlatego to założenie jest prawidłowe (najmniejsza liczba kierowców, która ukończyła wyścig F1 - 3: GP Monako w 1996 roku).

Przed usunięciem wartości NaN, kompletny zbiór danych przygotowanych do uczenia składa się z 3059 wierszy. Po selekcji, liczba wierszy to 540.

## Modele
W projekcie zastosowano 4 modele:
1. Rozpoznawanie obrazu
	1. sieci neuronowe - różne konfiguracje warstw ukrytych
2. Przewidywanie wyniku mistrzostw na podstawie 4 cech
	1. regresja logistyczna - z regularyzacją oraz bez
	

### Rozpoznawanie obrazu

Dla problemu rozpoznawania obrazu zastosowano 2 modele sieci neuronowych - jeden prostszy, a drugi bardziej zaawansowany, składający się z większej ilości warstw konwolucyjnych. Modele prezentują się następująco:

```python
def create_simple_model():  
    model = Sequential([  
        Conv2D(32, (3, 3), activation='relu', input_shape=(250, 250, 3)),  
        MaxPooling2D((2, 2)),  
        Conv2D(32, (3, 3), activation='relu'),  
        MaxPooling2D((2, 2)),  
        Flatten(),  
        Dense(128, activation='relu'),  
        Dropout(0.5),  
        Dense(number_of_classes, activation='softmax')
    ])  
      
    model.compile(loss=CategoricalCrossentropy(), optimizer="adam")
      
    return model  
  
def create_advanced_model():  
    model = Sequential([  
    Conv2D(32, (3, 3), activation='relu', input_shape=(250, 250, 3)),  
    MaxPooling2D((2, 2)),  
    Conv2D(64, (3, 3), activation='relu'),  
    MaxPooling2D((2, 2)),  
    Conv2D(128, (3, 3), activation='relu'),  
    MaxPooling2D((2, 2)),  
    Flatten(),  
    Dense(256, activation='relu'),  
    Dropout(0.5),  
    Dense(number_of_classes, activation='softmax')  
    ])  
      
    model.compile(loss=CategoricalCrossentropy(), optimizer="adam")
      
    return model
```

## Ewaluacje

Do ewaluacji wykorzystano metryki *accuracy, precision, recall* oraz *F-score*.

### Rozpoznawanie obrazu

Dla wielkości wsadu 32 oraz dla 3 epok, otrzymano następujące rezultaty:

| **Model**                  | **Accuracy** | **Precision** | **Recall** | **F-score** |
| -------------------------- | ------------ | ------------- | ---------- | ----------- |
| Prosta sieć neuronowa      | 0.9147       | 0.9879        | 0.8747     | 0.9278      |
| Rozbudowana sieć neuronowa | 0.9813       | 0.9813        | 0.9813     | 0.9813      |

### Problem regresji

## Wnioski

### Rozpoznawanie obrazu

Dla rozpoznawania obrazu, zastosowanie bardziej rozbudowanego modelu pozwoliło na zwiększenie dokładności kosztem dłuższego czasu uczenia. Dla tak prostego zbioru danych (niezbyt złożona grafika na białym tle), lepiej wybrać pierwszy model i zwiększyć liczbę epok.

Co ciekawe, podczas dostosowywania modelu, dla pewnej konfiguracji, _confusion matrix_ wyglądało następująco: ![confusion_matrix.png](confusion_matrix.png)

Praktycznie wszystkie zespoły były "odgadywane" prawidłowo, oprócz zespołu Alpine - ten potrafił być mylony z zespołem Williamsa. Spójrzmy na loga obu tych zespołów:

![alpine_williams.png](alpine_williams.png)

Oba z nich mają niebieski kolor oraz ostre kąty, co tłumaczy pomyłki modelu dla tych dwóch zespołów. Po zastosowaniu poprawek co do definicji modelu, mylenie tych dwóch zespołów zanikło.