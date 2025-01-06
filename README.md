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
Dane pochodzą ze strony kaggle.com: https://www.kaggle.com/datasets/cjgdev/formula-1-race-data-19502017

Aby przygotować dane do uczenia, należało:
1. Odrzucić kolumny, które nie wnoszą wartościowych informacji (takich jak numer kierowcy, liczba punktów, czas wyścigu itd.),
2. "Tabele" *races* oraz *results* należało połączyć za pomocą kolumny *raceId*, aby określić, z którego roku pochodzi dany rezultat. W ten sposób można policzyć wszystkie statystyki w każdym z sezonów osobno, ponieważ kierowcy mogą brać udział w więcej niż jednym sezonie. Za pomocą pętli for, która inkrementuje od 1950 do 2018, wybieram tylko te wiersze, które wartość *year* mają równą ze zmienną sterującą.
3. Zliczenie średniej z pozycji na starcie oraz pozycji na mecie, a następnie pogrupowanie tych wyników według unikalnego id kierowcy *driverId*,
6. Aby zliczyć % zwycięstw w sezonie, należało zliczyć wszystkie wyścigi w sezonie na podstawie kolumny *year*. Następnie, dla każdego kierowcy (*driverId*) zsumować kolumnę *positionOrder* w danym sezonie, jeżeli wartość wiersza = 1 (1. miejsce na mecie - zwycięstwo w wyścigu). Taką sumę dzielę przez liczbę wyścigów i otrzymuję % zwycięstw w sezonie dla każdego kierowcy.
7. Aby zliczyć % przejechanych okrążeń w sezonie, należało zliczyć wszystkie okrążenia znajdujące się w *lapTimes* dla każdego kierowcy. Liczba okrążeń składająca się na cały wyścig niestety nie jest przechowywana w tabeli *races*, dlatego zakładamy, że całkowita liczba okrążeń w wyścigu to wartość funkcji max z wszystkich zarejestrowanych okrążeń dla danego wyścigu - w całej historii Formuły 1 nie zdarzyło się, aby nikt nie ukończył wyścigu, dlatego to założenie jest prawidłowe (najmniejsza liczba kierowców, która ukończyła wyścig F1 - 3: GP Monako w 1996 roku).

Przed usunięciem wartości NaN, kompletny zbiór danych przygotowanych do uczenia składa się z 3059 wierszy. Po selekcji, liczba wierszy to 540.

## Modele
W projekcie zastosowano 4 modele - 2 modele do problemu klasyfikacji, oraz 2 modele do problemu regresji:
1. Problem klasyfikacji - rozpoznawanie obrazu
	1. sieci neuronowe
	2. regresja logistyczna
2. Problem regresji - przewidywanie wyniku mistrzostw na podstawie 4 cech
	1. regresja liniowa
	2. regresja wielomianowa

