import plotly.graph_objects as go
import numpy as np
from numpy import pi, sin, cos
import re
import json

dict_names = {}
with open('c:\\data\\general\\countries_dict.txt',
          encoding='utf-8') as f:
    data = f.read()
js = json.loads(data)
for k,v in js.items():
    for x in v:
        dict_names[x] = k
def clean_country_series(series):
    series = series.map(dict_names)
    return series

def point_sphere(lon, lat):
    #associate the cartesian coords (x, y, z) to a point on the  globe of given lon and lat
    #lon longitude
    #lat latitude
    lon = lon*pi/180
    lat = lat*pi/180
    x = cos(lon) * cos(lat)
    y = sin(lon) * cos(lat)
    z = sin(lat)
    return np.array([x, y, z])

def shortest_path(A=[100, 45], B=[-50, -25], dir=-1, n=100):
    #Spherical "linear" interpolation
    """
    A=[lonA, latA] lon lat given in degrees; lon in  (-180, 180], lat in ([-90, 90])
    B=[lonB, latB]
    returns n points on the great circle of the globe that passes through the  points A, B
    #represented by lon and lat
    #if dir=1 it returns the shortest path; for dir=-1 the complement of the shortest path
    """
    As = point_sphere(A[0], A[1])
    Bs = point_sphere(B[0], B[1])
    alpha = np.arccos(np.dot(As,Bs)) if dir==1 else  2*pi-np.arccos(np.dot(As,Bs))
    if abs(alpha) < 1e-6 or abs(alpha-2*pi)<1e-6:
        return A
    else:
        t = np.linspace(0, 1, n)
        P = sin((1 - t)*alpha)
        Q = sin(t*alpha)
        #pts records the cartesian coordinates of the points on the chosen path
        pts =  np.array([a*As + b*Bs for (a, b) in zip(P,Q)])/sin(alpha)
        #convert cartesian coords to lons and lats to be recognized by go.Scattergeo
        lons = 180*np.arctan2(pts[:, 1], pts[:, 0])/pi
        lats = 180*np.arctan(pts[:, 2]/np.sqrt(pts[:, 0]**2+pts[:,1]**2))/pi
        return lons, lats

def remove_asterisc(x):
    x = re.sub(r"[*]", '', x)
    x = re.sub("   ", "", x)
    return x

dict_mex_names = {
    'Aguascalientes' : 'Aguascalientes',
    'Baja California Sur' : 'Baja California Sur',
    'Baja California' : 'Baja California',
    'Campeche' : 'Campeche',
    'Chiapas' : 'Chiapas',
    'Chihuahua' : 'Chihuahua',
    'Coahuila' : 'Coahuila',
    'Colima' : 'Colima',
    'Distrito Federal ' : 'Ciudad de México',
    'Durango' : 'Durango',
    'Estado de Mexico' : 'Estado de México',
    'Guanajuato' : 'Guanajuato',
    'Guerrero' : 'Guerrero',
    'Hidalgo' : 'Hidalgo',
    'Jalisco' : 'Jalisco',
    'Michoacan' : 'Michoacán',
    'Morelos' : 'Morelos',
    'Nayarit' : 'Nayarit',
    'Nuevo Leon' : 'Nuevo León',
    'Oaxaca' : 'Oaxaca',
    'Puebla' : 'Puebla',
    'Queretaro' : 'Querétaro',
    'Quintana Roo' : 'Quintana Roo',
    'San Luis Potosi' : 'San Luis Potosí',
    'Sinaloa' : 'Sinaloa',
    'Sonora' : 'Sonora',
    'Tabasco' : 'Tabasco',
    'Tamaulipas' : 'Tamaulipas',
    'Tlaxcala' : 'Tlaxcala',
    'Veracruz' : 'Veracruz',
    'Colima2011.xlsxF' : 'Colima',
    'Distrito Federal' : 'Ciudad de México',
    'Durango2011.xlsxF' : "Durango",
    'Edo Mex' : 'Estado de México',
    'Oaxaca2011f' : "Oaxaca",
    'Quereataro2011f' : 'Querétaro',
    'Yucatan' : 'Yucatán',
    'Zacatecas' : 'Zacatecas',
    'NuevoLeon' : 'Nuevo León',
    'aguascalientes' : 'Aguascalientes',
    'baja california' : 'Baja California',
    'baja california sur' : 'Baja California Sur',
    'campeche' : 'Campeche',
    'chiapas' : 'Chiapas',
    'chihuahua' : 'Chihuahua',
    'ciudad de mxico' : 'Ciudad de México',
    'coahuila' : 'Coahuila',
    'colima' : 'Colima',
    'durango' : 'Durango',
    'estado de mxico' : 'Estado de México',
    'guanajuato' : 'Guanajuato',
    'guerrero' : 'Guerrero',
    'hidalgo' : 'Hidalgo',
    'jalisco' : 'Jalisco',
    'michoacn' : 'Michoacán',
    'morelos' : 'Morelos',
    'nayarit' : 'Nayarit',
    'nuevo len' : 'Nuevo León',
    'puebla' : 'Puebla',
    'quertaro' : 'Querétaro',
    'quintana roo' : 'Quintana Roo',
    'san luis potos' : 'San Luis Potosí',
    'sinaloa' : 'Sinaloa',
    'sonora' : 'Sonora',
    'tabasco' : 'Tabasco',
    'tamaulipas' : 'Tamaulipas',
    'tlaxcala' : 'Tlaxcala',
    'veracruz' : 'Veracruz',
    'yucatn' : 'Yucatán',
    'zacatecas' : 'Zacatecas',
    'san_luis_potosi' : 'San Luis Potosí',
    'oaxaca' : 'Oaxaca',
    'Ciudad de México' : 'Ciudad de México',
    'Estado de México' : 'Estado de México',
    'Hildalgo' : 'Hidalgo',
    'Michoacán' : 'Michoacán',
    'Nuevo León' : 'Nuevo León',
    'Nuevo leon' : 'Nuevo León',
    'Querétaro' : 'Querétaro',
    'San Luis Potosí' : 'San Luis Potosí',
    'Yucatán' : 'Yucatán',
    'Coahuila de Zaragoza' : 'Coahuila',
    'Veracruz de Ignacio de la Llave' : 'Veracruz',
    'Michoacán de Ocampo' : 'Michoacán',
    'México' : 'Estado de México',
    'Tamaupilas' : 'Tamaulipas',
    'Zacatec' : 'Zacatecas',
    'Mexico city' : 'Ciudad de México',
    'Mexico' : 'Estado de México'
}

css_colors = open("c:\\data\\general\\css_colors.txt", "r")
css_colors = css_colors.read()
css_colors = css_colors.split(", ")

austria_nighbours = ["Germany", "Czechia", "Switzerland", "Italy", "Slovakia", "Hungary", "Slovenia"]

def find_outliers_iqr(df):

   q1=df.quantile(0.25)
   q3=df.quantile(0.75)
   IQR=q3-q1
   outliers = df[((df<(q1-5*IQR)) | (df>(q3+5*IQR)))]

   return outliers