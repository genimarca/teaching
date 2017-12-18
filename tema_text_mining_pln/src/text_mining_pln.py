#!/usr/bin/python3
# *-* coding utf-8 *-*
'''
Created on 15 dic. 2017

@author: Eugenio Martínez Cámara <emcamara@decsai.ugr.es>
'''

import re
from nltk import bigrams
from nltk import trigrams
from nltk import sent_tokenize
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from urllib import request

"""Sección de variables globales

    URL_LIBRO: String con la ruta en donde se encuentre el libro 
    a descargar en texto plano. En el ejemplo se va a utilizar el 
    Quijote.
"""

URL_LIBRO = "http://www.gutenberg.org/cache/epub/2000/pg2000.txt"



def download_data():
    """Método para descargar los datos a procesar
    
    Return:
        Un String con el contenido del libro.
    """
    
    file_book = request.urlopen(URL_LIBRO)
    raw_book = file_book.read().decode("utf-8")
    return raw_book
    
def segmentar(text_data):
    """Identifica las oraciones que hay en el texto de entrada
    
    Returns:
        Una lista de oraciones
    """
    return sent_tokenize(text_data, language="spanish")

def tokenize(sents):
    """Identifica los tokens del las oraciones de entrada
    
    Returns:
        Una lista de oraciones. Cada oración es una lista de tokens
    """
    tokenizer = TreebankWordTokenizer()
    
    sent_tokens = [tokenizer.tokenize(sent) for sent in sents]
    
    return sent_tokens


def ejemplo_segmentacion(sents):
    """Muestra las oraciones extraídas
    
    Args:
        sents: Lista de oraciones
    """
    print("5 oraciones del libro descargado: %s" % "\n".join(sents_book[500:505]))
    n_sents = len(sents_book)
    print("\nNº. de oraciones: %d" % n_sents)
    

def ejemplo_tokenizar(sents):
    """Muestra palabras/tokens
    
    Args:
        tokens: lista de listas de strings
    """
    
    print("Tokens de 5 oraciones: %s" % "\n".join(map(lambda x:", ".join(x), sents[500:505])))
    n_words = 0
    for sent in sents:
        n_words += len(sent)
    print("\nNº. de palabras: %d" % n_words)
    

def calculo_frecuencias(bag_of_words):
    """Calcula frecuencias de las palabras y muestra una gráfica con las más frecuentes
    
    Args:
        bag_of_words: lista de strings
    """
    freq_dist = FreqDist(bag_of_words)
    print("Nº. objetos: %d"%freq_dist.N())
    print("Nº. objetos únicos: %d"%freq_dist.B())
    print("El objeto más frecuente es: %s" % str(freq_dist.max()))
    freq_dist.plot(50)

def rm_stopwods(bag_of_words):
    """Elimina las stopwords del texto de entrada
    
    Args:
        sent_words_book: lista de strings.
    
    Returns:
        Una lista de listas de oraciones por palabras sin incluir las 
        palabras vacías.
    """
    spanish_stopwords = stopwords.words("spanish")
    no_stop_words = [word for word in bag_of_words if word not in spanish_stopwords]
    return no_stop_words

def rm_puntuation(bag_of_words):
    """Elimina los signos de puntuación.
    
    Args:
        bag_of_words: lista de strings.
        
    """
    re_puntuation = re.compile(r"""[\.,":;]""")
    bag_of_words_no_puntation = [word for word in bag_of_words if re_puntuation.fullmatch(word) is None]
    return bag_of_words_no_puntation

if __name__ == '__main__':
    
    #Obtener datos
    raw_book = download_data()
    print("Descargado el libro")
    
    print("Parte del texto descargado\n")
    print(raw_book[500:520])
    print("---------------------------")
    
    print("Segmentar\n")
    sents_book = segmentar(raw_book)
    ejemplo_segmentacion(sents_book)
    print("---------------------------")
    
    print("Tokenizar\n")
    sents_words_book = tokenize(sents_book)
    ejemplo_tokenizar(sents_words_book)
    print("---------------------------")
    
    bag_of_words = [word.lower() for sent in sents_words_book for word in sent]
    
    print("Frecuencias\n")
    calculo_frecuencias(bag_of_words)
    print("---------------------------")
    
    
    print("Stopper\n")
    bag_of_words_no_stopper = rm_stopwods(bag_of_words)
    calculo_frecuencias(bag_of_words_no_stopper)
    print("---------------------------")
    
    print("No signos de puntuación\n")
    bag_of_words_no_stopper_no_puntuacion = rm_puntuation(bag_of_words_no_stopper)
    calculo_frecuencias(bag_of_words_no_stopper_no_puntuacion)
    print("---------------------------")
    
    print("Unigramas/Bigramas/Trigramas\n")
    
    print("Unigramas: %s" % ", ".join(bag_of_words_no_stopper_no_puntuacion[500:510]))
    print("---------------------------")
    print("-Bigramas")
    bag_of_bigrams = list(bigrams(bag_of_words_no_stopper_no_puntuacion))
    calculo_frecuencias(bag_of_bigrams)
    print("---------------------------")
    print("-Trigramas")
    bag_of_trigrams = trigrams(bag_of_words_no_stopper_no_puntuacion)
    calculo_frecuencias(bag_of_trigrams)
    print("---------------------------")
    
