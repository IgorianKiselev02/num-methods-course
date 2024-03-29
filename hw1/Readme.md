## Система

Все запуски проводились на MacBook Pro M1 MAX 64GB RAM

Использовался OpenMp из cellar llvm 17.0.6_1


## Идея для эксперимента

* Во-первых, проверить насколько удачно реализован многопоточный алгоритм (в сравнении с результатами в книге, а конкретно с коэфициентом ускорения для 8ми потоков) и какой прирост по скорости по сравнению с базовой версией я получу на разном количестве потоков. Ожидаю увеличение скорости работы алгоритма на 2-ух, 4-х и 8-и потоках, и снижение на 16-ти по сравнению с 8-ю. 1 поток должен в свою очередь работу программы немного замедлить. Данный факт объясняется не столько математикой, сколько работой потоков - для 16-ти возникает слишком много дополнительных затрат и действий по сравнеии с 8-ми потоками, так же для 1го потока мы выполним дополнительные действия по сравнению с базовым алгоритмом. С точки зрения математики мой теоретический результат для 16-ти потоков также можно объяснить - в таком случае у нас может возникать ситуация, когда потоков больше чем блоков, в связи с чем потоки простаивают.

* Во-вторых, проверить насколько это соотношение работает для "частично-долгих" функций. Я рассматриваю 2 набора функций: один набор из примера в книге, другой немного отличается: функция f долго выполняется на некотрых входах (поэтому я и называю её "частично-долгой"). Долгое вычисление иммитируется за счёт банального сна потока - я решил, что это надёжнее, чем выдумывать огромные формулы. Я ожидаю, что соотношение коэффициентов ускорения у "частично-долгих" функций будет быстрее затухать, так как скорость вычисления 1го результата вызова функции нельзя ускорить за счёт многопоточности, если конечно сама функция не реализует его. Поэтому предел больше которого мы не можем ускорить код за счёт распараллеливания увеличивается как раз на время выполнения этих функций.

Функции из примера в книге я взял, так как с их помощью я могу сравнить скорость работы параллельного и обычного алгоритмов, а также сравнить коэффициент ускорения с аворским. Более того в процессе написания самих алгоритмов я запускался на тех же параметрах, что указали авторы (запуски для эксперементов используют другой набор параметров), чтобы иметь возможность себя проверить. Однако также стоит отметить, что для первого эксперимента подошли бы любые быстро работающие функции с нужной нам областью определния. Для второго эксперимента я модифицировал именно функции из первого эксперимента, так как мне нужно сравнить получившееся соотношение именно с функцией именно из первого эксперимента.

## Тестовые данные

Для каждого эксперимента происходит 20 запусков, отсекается по 2 крайних случая с каждой стороны и берётся среднее на основе оставшихся данных.

Данные при запуске:

* Размер сети равен 3000

* Количество блоков равно 150

* Размер блока равен 20

* Значение епсилон выбрано 0.15 (Авторы книги использовали 0.1, я выбрал чуть более большее значение для ускорения времени работы. Соответсвенно чем больше епсилон, тем раньше алгоритм сойдётся.)

## Практические результаты

### Базовый алгоритм

|                    | Функция из примера | "частично-долгая" функция |
|--------------------|--------------------|---------------------------|
|   Средднее время   |       40.4848      |          55.4355          |
| Максимальное время |       40.6904      |          55.623           |
| Минимальное время  |       40.0837      |          55.3146          |

* Функция из примера

  Все результаты запусков отсортированные по возрастанию (первые и последние 2 результата не учитываются при вычислении среднего):

  40.0837 40.0945 40.2395 40.2585 40.2669 40.3522 40.4559 40.5057 40.5212 40.5345 40.5392 40.5408 40.5417 40.5522 40.588 40.5911 40.6177 40.6524 40.6614 40.6904

* "частично-долгая" функция

  Все результаты запусков отсортированные по возрастанию (первые и последние 2 результата не учитываются при вычислении среднего):

  55.3146 55.3761 55.3777 55.3849 55.3871 55.4099 55.4144 55.4164 55.4192 55.4229 55.4325 55.4447 55.4451 55.4629 55.4793 55.4855 55.4908 55.4944 55.5878 55.623

### Параллельный алгоритм

|               |                | Функция из примера |                   |           |                |   "частично-долгая"|функция            |           |
|---------------|----------------|--------------------|-------------------|-----------|----------------|--------------------|-------------------|-----------|
| Число потоков | Средднее время | Максимальное время | Минимальное время | ускорение | Средднее время | Максимальное время | Минимальное время | ускорение |
|        1      |     49.7170    |       50.2089      |       49.1717     |  0.8143   |     63.7167    |       64.4047      |       63.3245     |  0.8700   |
|        2      |     26.7202    |       27.2905      |       26.2614     |  1.5151   |     40.4755    |       43.6248      |       40.1858     |  1.3696   |
|        4      |     15.6353    |       15.8375      |       15.5989     |  2.5563   |     29.6504    |      109.4700      |       29.2782     |  1.8696   |
|        8      |     11.5947    |       11.9851      |       11.1307     |  3.4917   |     25.2914    |       25.6251      |       25.0149     |  2.1919   |
|        16     |     12.8234    |       13.1978      |       12.6081     |  3.1571   |     26.0499    |       26.1898      |       25.8544     |  2.1281   |


* Функция из примера

  Все результаты запусков отсортированные по возрастанию (первые и последние 2 результата не учитываются при вычислении среднего):

  | Число потоков |                            Время работы по возрастанию                          |
  |---------------|---------------------------------------------------------------------------------|
  |       1       | 49.1717 49.2088 49.2932 49.4674 49.5152 49.5456 49.5479 49.6422 49.6638 49.6843 49.7394 49.7436 49.7442 49.8292 49.9286 49.9386 50.044 50.1441 50.1829 50.2089 |
  |       2       | 26.2614 26.3082 26.353 26.357 26.3653 26.3662 26.443 26.4671 26.5186 26.7149 26.7879 26.8758 26.9024 27.012 27.0223 27.0775 27.0884 27.1726 27.1774 27.2905 |
  |       4       | 15.5989 15.6 15.6096 15.6134 15.6162 15.6179 15.6199 15.629 15.6293 15.6303 15.6341 15.6366 15.641 15.649 15.6502 15.6573 15.6618 15.6699 15.7163 15.8375|
  |       8       | 11.1307 11.1329 11.1871 11.2422 11.3558 11.4299 11.4305 11.4842 11.4863 11.5011 11.642 11.6968 11.785 11.8013 11.8462 11.8647 11.8728 11.8888 11.9171 11.9851 |
  |       16      | 12.6081 12.6378 12.6824 12.6954 12.714 12.7489 12.7943 12.8101 12.8103 12.8393 12.8428 12.8491 12.8552 12.8657 12.8889 12.894 12.9019 12.9815 13.0062 13.1978 |

* "частично-долгая" функция

  Все результаты запусков отсортированные по возрастанию (первые и последние 2 результата не учитываются при вычислении среднего):

  | Число потоков |                            Время работы по возрастанию                          |
  |---------------|---------------------------------------------------------------------------------|
  |       1       | 63.3245 63.3351 63.3896 63.4421 63.5291 63.5408 63.5516 63.605 63.6111 63.6154 63.692 63.723 63.7376 63.8747 63.8807 63.9859 64.0814 64.207 64.2924 64.4047 |
  |       2       | 40.1858 40.2659 40.3011 40.3272 40.3377 40.3442 40.3462 40.3473 40.3496 40.3708 40.4574 40.4916 40.5145 40.5345 40.5979 40.7526 40.7607 40.7752 40.954 43.6248 |
  |       4       | 29.2782 29.363 29.3842 29.3878 29.4659 29.4699 29.4888 29.5402 29.5636 29.5647 29.5841 29.594 29.6094 29.6836 29.9599 29.9767 29.9822 30.1519 31.7451 109.47 |
  |       8       | 25.0149 25.0381 25.0687 25.0836 25.1092 25.129 25.1538 25.2776 25.3183 25.3474 25.3484 25.3623 25.3721 25.379 25.3881 25.4361 25.4438 25.4446 25.5403 25.6251 |
  |       16      | 25.8544 25.8989 25.9427 25.9632 25.9634 25.9838 25.9901 25.9959 26.0486 26.0489 26.0567 26.0629 26.1145 26.1202 26.1206 26.123 26.1283 26.1358 26.139 26.1898 |

  Тут появилось число 109 - ноутбук успел уснуть, поэтому такой вот выброс получился...
## Выводы

* В 1ом пункте практический прирост совпал с моими ожиданиями. Говоря про соотношение с базовым алгоритмом - на 8-ми потоках у авторов получился прирост в 4.52 раза, у меня же 3.4917. Разницу можно объяснить за счёт другого епсилона, разницы в системе, а также черезмерным, на мой взгляд, выделением памяти. Думаю, что уменьшить работу с памятью в моей версии алгоритма, то его можно ускорить.

* В 2ом эксперементе практическое соотношение коэффициентов ускорения также совпало с моими ожиданиями. Более того, можно заметить следующий факт: абсолтная разница между средним временем выполнения для разных потоков для обоих наборов функций очень близка. Также можно заметить, что для одного числа потоков разность среднего времени выполнения для 2ух случаев также всегда около 14 секунд (Однако это число не совпадает с произведением числа итераций на количество вызовов и время сна. Веротянее всего это объясняется задержкой "пробуждения" потока). Это подтверждает мои догадки про то, что время выполнения параллельного алгоритма увеличивается ровно на время выполнения функций.