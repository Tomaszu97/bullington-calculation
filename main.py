# from matplotlib import rcParams
# rcParams['font.family'] = 'comic sans ms'
import matplotlib.pyplot as plt
import numpy as np
from time import sleep
import math
import scipy.integrate as integrate
import scipy.special as special
import cmath

#all units are meters
class Bullington():
    def __init__(self):
        self.TX = None, None
        self.O1 = None, None  #obstacle 1
        self.O2 = None, None
        self.RX = None, None
        self.BEQ = None, None #Bullington EQuiavlent
        self.wavelength = 1.3 # 1.3m ~ 230 MHz
        plt.style.use('dark_background')
        plt.figure(figsize=(16.2,10.8))
        plt.tight_layout()
        plt.grid()

        # axes = plt.gca()
        # axes.set_xlim([0,10000])
        # axes.set_ylim([-10,1000])


    def set_apexes(self, TX=(0,100), O1=(2500, 500), O2=(7500, 500), RX=(10000,100)):
        self.TX = TX
        self.O1 = O1
        self.O2 = O2
        self.RX = RX

    def set_frequency(self, frequency_MHz):
        self.wavelength = 300/frequency_MHz
    

    def load_terrain(self, terrain_filename):
        terrain_file = open(terrain_filename, encoding='utf-8')
        terrain_samples = []
        for line in terrain_file:
            line = line.strip()
            x, y = int(line.split(',')[0]) , float(line.split(',')[1])
            terrain_samples.append([x,y])
        self.terrain = np.array(terrain_samples)

    def plot_save(self, filename='out.png'):
        plt.savefig(filename)

    def plot_clear(self):
        plt.clf()

    def plot_terrain(self):
        tempterrain = self.terrain[:,:]        
        plt.fill_between(x=tempterrain[:,0], y1=tempterrain[:,1], y2=np.amin(tempterrain[:,1]), facecolor='#5500FF', alpha = 0.7)

    def plot_apexes(self):
        plt.scatter(self.TX[0], self.TX[1], c='#0066FF', zorder=10)
        plt.text(self.TX[0], self.TX[1],' TX', c='#0066FF', fontsize = 14, weight='bold')
        plt.scatter(self.RX[0], self.RX[1], c='#11CC11', zorder=10)
        plt.text(self.RX[0], self.RX[1],' RX', c='#11CC11', fontsize = 14, weight='bold')
        plt.scatter(self.O1[0], self.O1[1], c='#AA3333', zorder=10)
        plt.text(self.O1[0], self.O1[1],' O1', c='#AA3333', fontsize = 14, weight='bold')
        plt.scatter(self.O2[0], self.O2[1], c='#AA3300', zorder=10)
        plt.text(self.O2[0], self.O2[1],' O2', c='#AA3333', fontsize = 14, weight='bold')
        plt.scatter(self.BEQ[0], self.BEQ[1], c='#FF3300', zorder=10)
        plt.text(self.BEQ[0], self.BEQ[1],' BEQ', c='#FF3300', fontsize = 14, weight='bold')


    def plot_lines(self):
        plt.plot((self.TX[0], self.BEQ[0]),(self.TX[1], self.BEQ[1]), c='#FF9999', linestyle='dashed')
        plt.plot((self.RX[0], self.BEQ[0]),(self.RX[1], self.BEQ[1]), c='#FF9999', linestyle='dashed')
        plt.plot((self.TX[0], self.RX[0]),(self.TX[1], self.RX[1]), c='#9999FF', linestyle='dashed')

    

    def calc_bullington_equivalent(self):
        line1 = (self.TX, self.O1)
        line2 = (self.RX, self.O2)
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]
        div = det(xdiff, ydiff)
        if div == 0:
            raise Exception('lines do not intersect')
        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        
        if x > self.O1[0] and x < self.O2[0]:
            self.BEQ = (x, y)
        else:
            print('################################################################')
            print('BŁĄD! Szczyt przeszkody zastępczej nie mieści się pomiędzy O1 i O2!\nSprawdż pozycje RX,TX,O1,O2!')
            print('################################################################')
            exit()

    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def place_object(self, position=0, rel_height=0, type='TX'):
        pos = self.find_nearest(self.terrain[:,0], position)
        for p, h in self.terrain:
            if p == pos:
                abs_height = h + rel_height
                break

        if type == 'TX':
            self.TX = (pos, abs_height)
        elif type == 'RX':
            self.RX = (pos, abs_height)
        elif type == 'O1':
            self.O1 = (pos, abs_height)
        elif type == 'O2':
            self.O2 = (pos, abs_height)
        else:
            raise Exception(f'{type}? Niepoprawna nazwa obiektu! Wybierz TX, RX, O1 lub O2')

    
    def calc_geometry(self):
        a = math.sqrt( ((self.BEQ[0]-self.TX[0])**2) + ((self.BEQ[1]-self.TX[1])**2) )
        b = math.sqrt( ((self.BEQ[0]-self.RX[0])**2) + ((self.BEQ[1]-self.RX[1])**2) )
        d = math.sqrt( ((self.RX[0]-self.TX[0])**2) + ((self.RX[1]-self.TX[1])**2) )
        
        s = (a+b+d)/2
        area = math.sqrt(s * (s - a) * (s - b) * (s - d))
        h = 2*area/d

        d1 = math.sqrt((a**2) - (h**2))
        d2 = d - d1
        self.a = a
        self.b = b
        self.h = h
        self.d1 = d1
        self.d2 = d2


    def calc_v(self):
        self.v = self.h * math.sqrt( 2*(self.d1+self.d2)/(self.wavelength*self.d1*self.d2) )

    def calc_fresnel_rn(self, n=1):
        return math.sqrt((n*self.wavelength*self.d1*self.d2)/(self.d1+self.d2))

    def calc_fresnel_r123(self):
        self.r1 = math.sqrt((1*self.wavelength*self.d1*self.d2)/(self.d1+self.d2))
        self.r2 = math.sqrt((2*self.wavelength*self.d1*self.d2)/(self.d1+self.d2))
        self.r3 = math.sqrt((3*self.wavelength*self.d1*self.d2)/(self.d1+self.d2))

    def calc_C_S_F(self):
        self.C = 0.5 - integrate.quad(lambda t: math.cos( (math.pi/2)*(t**2) ) , 0, self.v)[0]
        self.S = 0.5 - integrate.quad(lambda t: math.sin( (math.pi/2)*(t**2) ) , 0, self.v)[0]
        self.F = ((1+1j)/2)*complex(self.C, -self.S)

    def calc_all(self):
        self.calc_bullington_equivalent()
        self.calc_geometry()
        self.calc_v()
        self.calc_fresnel_r123()
        self.calc_C_S_F()

    def print_all_data(self):
        print(f'Częstotliwość: {round(300/self.wavelength,0)} MHz')
        print(f'Pozycja nadajnika: {self.TX[0]} m')
        print(f'Wysokość nadajnika n.p.m.: {self.TX[1]} m')
        print(f'Pozycja odbiornika: {self.RX[0]} m')
        print(f'Wysokość odbiornika n.p.m.: {self.RX[1]} m')
        print(f'Pozycja przeszkody 1: {self.O1[0]} m')
        print(f'Wysokość przeszkody 1 n.p.m.: {self.O1[1]} m')
        print(f'Pozycja przeszkody 2: {self.O2[0]} m')
        print(f'Wysokość przeszkody 2 n.p.m.: {self.O2[1]} m')
        print(f'Pozycja przeszkody zastępczej: {round(self.BEQ[0],1)} m')
        print(f'Wysokość przeszkody zastępczej n.p.m.: {round(self.BEQ[1],1)} m')
        print(f'a={round(self.a,2)} m\nb={round(self.b,2)} m\nd1={round(self.d1,2)} m\nd2={round(self.d2,2)} m\nh={round(self.h,2)} m')
        print(f'Parametr v={round(self.v,2)}')
        print('Promienie stref fresnela dla odległości d1 od TX do RX:')
        print(f'Promień 1 strefy r1={round(self.r1,2)} m')
        print(f'Promień 2 strefy r2={round(self.r2,2)} m')
        print(f'Promień 3 strefy r3={round(self.r3,2)} m')
        print(f'C(v)={round(self.C,3)}\nS(v)={round(self.S,3)}\nRe(F(v))={round(self.F.real,3)}\nIm(F(v))={round(self.F.imag,3)}\n|F(v)|={round(abs(self.F),3)}')
        print(f'F[dB]={round(-20*math.log10(abs(self.F)),3)}')

    def make_plot_image(self):
        self.plot_terrain()
        self.plot_apexes()
        self.plot_lines()
        self.plot_save()

print('################################################################')
print("""                       .,,uod8B8bou,,.
              ..,uod8BBBBBBBBBBBBBBBBRPFT?l!i:.
         ,=m8BBBBBBBBBBBBBBBRPFT?!||||||||||||||
         !...:!TVBBBRPFT||||||||||!!^^""'   ||||
         !.......:!?|||||!!^^""'      .     ||||
         !.........||||     ___/_____._._RX ||||
         !.........||||       / .   .  .    ||||
         !.........||||      / . . .   .    ||||
         !.........||||     / .  .          ||||
         !.........||||    . .    .         ||||
         !.........||||   . .               ||||
         `.........||||   .                ,||||
          .;.......|||| TX            _.-!!|||||
   .,uodWBBBBb.....||||       _.-!!|||||||||!:'
!YBBBBBBBBBBBBBBb..!|||:..-!!|||||||!iof68BBBBBb....
!..YBBBBBBBBBBBBBBb!!||||||||!iof68BBBBBBRPFT?!::   `.
!....YBBBBBBBBBBBBBBbaaitf68BBBBBBRPFT?!:::::::::     `.
!......YBBBBBBBBBBBBBBBBBBBRPFT?!::::::;:!^"`;:::       `.
!........YBBBBBBBBBBRPFT?!::::::::::^''...::::::;         iBBbo.
`..........YBRPFT?!::::::::::::::::::::::::;iof68bo.      WBBBBbo.
  `..........:::::::::::::::::::::::;iof688888888888b.     `YBBBP^'
    `........::::::::::::::::;iof688888888888888888888b.     `
      `......:::::::::;iof688888888888888888888888888888b.
        `....:::;iof688888888888888888888888888888888899fT!
          `..::!8888888888888888888888888888888899fT|!^"'
            `' !!988888888888888888888888899fT|!^"'
                `!!8888888888888888899fT|!^"'
                  `!988888888899fT|!^"'
                    `!9899fT|!^"'
                      `!^"'""")
print('################################################################')
print('Program do wyznaczania strat propagacyjnych metodą Bullingtona.')
print('Tomasz Jakubowski 2020')
print('################################################################')
bul = Bullington()




filename = input('Witaj w programie! Poniżej dostępne opcje:\npodaj nazwę pliku z przekrojem terenu\nzostaw puste aby załadować płaski teren (o długości 10000 m)\nkasina - przykładowy scenariusz\nnowadęba - przykładowy scenariusz\nrabka - przykładowy scenariusz\n')

try:
    if filename == 'kasina':
        bul.load_terrain('profil_kasinawielka_kluszkowce.csv')
        bul.place_object(5475, 0, type='TX')
        bul.place_object(27841, 0, type='RX')
        bul.place_object(15180, 0, type='O1')
        bul.place_object(19336, 0, type='O2')
        bul.set_frequency(230)
        bul.calc_all()
        bul.print_all_data()
        bul.make_plot_image()
        exit()
    elif filename == 'nowadęba':
        bul.load_terrain('profil_nowadeba_bilgoraj.csv')
        bul.place_object(0, 10, type='TX')
        bul.place_object(95000, 10, type='RX')
        bul.place_object(12477, 0, type='O1')
        bul.place_object(88750, 0, type='O2')
        bul.set_frequency(230)
        bul.calc_all()
        bul.print_all_data()
        bul.make_plot_image()
        exit()
    elif filename == 'rabka':
        bul.load_terrain('profil_rabkazdroj_nowysacz.csv')
        bul.place_object(11600, 0, type='TX')
        bul.place_object(42468, 0, type='RX')
        bul.place_object(22776, 0, type='O1')
        bul.place_object(34099, 0, type='O2')
        bul.set_frequency(230)
        bul.calc_all()
        bul.print_all_data()
        bul.make_plot_image()
        exit()
    else:
        bul.load_terrain(input)
except TypeError:
    print('Ładowanie płaskiego terenu')
    bul.load_terrain('flat.csv')

frequency = int(input('Podaj częstotliwość w MHz:\n'))
bul.set_frequency(frequency)

TXpos = int(input('Podaj pozycję nadajnika:\n'))
TXrelh = float(input('Podaj wysokość nadajnika n.p.t.:\n'))
bul.place_object(TXpos, TXrelh, type='TX')

RXpos = int(input('Podaj pozycję odbiornika:\n'))
RXrelh = float(input('Podaj wysokość odbiornika n.p.t.:\n'))
bul.place_object(RXpos, RXrelh, type='RX')

O1pos = int(input('Podaj pozycję przeszkody 1:\n'))
O1relh = float(input('Podaj wysokość przeszkody 1 n.p.t.:\n'))
bul.place_object(O1pos, O1relh, type='O1')

O2pos = int(input('Podaj pozycję przeszkody 2:\n'))
O2relh = float(input('Podaj wysokość przeszkody 2 n.p.t.:\n'))
bul.place_object(O2pos, O2relh, type='O2')

bul.calc_all()
bul.print_all_data()
bul.make_plot_image()


