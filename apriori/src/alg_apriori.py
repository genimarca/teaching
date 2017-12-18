'''
Created on 27 nov. 2017

@author: Eugenio Martínez Cámara <emcamara@decsai.ugr.es>
'''


def load_dataset():
    """Lectura en memoria del dataset
    
    Returns: Una lista de transacciones.
    """
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

def create_base_of_items_c0(data_base_transactions):
    """Generación del conjunto de items
    
    Returns: Lista con los items
    """
    items = []
    for transaction in data_base_transactions:
        for item in transaction:
            f_item = frozenset([item])
            if not f_item in items:
                items.append(frozenset([item])) #Truco para hacer más eficiente el cálculo del soporte
    items.sort(key=lambda x:list(x))
    return items

def create_lk(data_base_transactions, Ck, min_support):
    """Cálculo de los itemsets frecuentes.
    
    Args:
        data_base_transactions: Lista de transacciones
        Ck: lista de itemsets candidatos
        minSupport: Valor real que indica el soporte mínimo de los 
        conjuntos frecuentes (itermsets).
    """
    lk = {}
    soporte_unitario = 1/len(data_base_transactions)
    for transaction in data_base_transactions:
        for item in Ck:
            if item.issubset(transaction):
                #Uso del valor por defecto de get para ahorrar un if
                #Suma de 1/n_transacciones para evitar luego tener que hacer otro bucle para normalizar
                lk[item] = lk.get(item, 0) + soporte_unitario #Uso del valor por defecto de get para ahorrar un if
    lk = {item:support for item, support in lk.items() if support >= min_support}
    return lk

def create_ck(Lk_1, k):
    """Genera los conjuntos candidatos de longitud k
    
    Args:
        Lk_1: lista de conjuntos con soporte de longitud k-1
        k: Nueva longitud
    
    """
    ck_next = []
    n_itemsets = len(Lk_1)
    for i in range(n_itemsets):
        for j in range(i+1, n_itemsets):
            c1 = sorted(list(Lk_1[i])[:k-2])
            c2 = sorted(list(Lk_1[j])[:k-2])
            if c1 == c2: #Prefijos iguales
                ck_next.append(Lk_1[i] | Lk_1[j]) #Unión de conjuntos
    return ck_next
        
    
def apriori(database_transactions, min_support=0.5):
    """Generación de los conjuntos de items frecuentes
    
    Args:
        database_transactions: Lista de transacciones
    """
    
    #Generación del conjunto inicial de items
    c0 = create_base_of_items_c0(database_transactions)
    l1 = create_lk(database_transactions, c0, min_support)
    lk = [l1]
    k=2
    while len(lk[k-2]) > 0:
        ck = create_ck(list(lk[k-2].keys()), k)
        l1_next = create_lk(database_transactions, ck, min_support)
        lk.append(l1_next)
        k += 1
    return lk

def calculate_conf(lk, freq_item_set, h1, min_conf):
    """Generara aquellas reglas del conjunto de items frecuentes que
    supera el umbral de confianza
    """
    
    rules = []
    for consequent in h1:
        conf = lk[freq_item_set]/lk[freq_item_set-consequent]
        if conf >= min_conf:
            rules.append((freq_item_set-consequent, consequent, conf))
    return rules

def rules_from_consequent(lk, freq_item_set, h1, min_conf):
    
    n_consequents = len(h1)
    rules = []
    if len(freq_item_set) > (n_consequents + 1):
        ck_consequents = create_ck(h1, n_consequents+1)
        rules_consequents = calculate_conf(lk, ck_consequents, h1, min_conf)
        if len(rules_consequents) > 1:
            rules.append(rules_from_consequent(lk, freq_item_set, rules_consequents, min_conf))
        else:
            return rules_consequents

def generate_rules(lk, min_conf=0.7):
    """Genera reglas a partir del conjunto de items frecuentes
    
    """
    
    rule_list = []
    support_data = {freq_item_set:support for freq_item_sets in lk for freq_item_set, support in freq_item_sets.items()}
    for i in range(1, len(lk)): #Sólo tomamos los conjuntos de items de más de un elemento.
        for freq_item_set in lk[i]:
            h1 = [frozenset([item]) for item in freq_item_set]
            if i > 1:
                rules = rules_from_consequent(support_data, freq_item_set, h1, min_conf)
            else:
                rules = calculate_conf(support_data, freq_item_set, h1,min_conf)
            if rules:
                rule_list += rules
    return rule_list

def print_rules(rules):
    
    for rule in rules:
        print("{} ====> {}: {}".format(", ".join(map(str,rule[0])), ", ".join(map(str,rule[1])), rule[-1]))

if __name__ == '__main__':
    
    transactions_database = load_dataset()
    print(transactions_database)
    ck = create_base_of_items_c0(transactions_database)
    print(ck)
    lk = create_lk(transactions_database, ck, 0.3)
    print(lk)
    
    
    lk = apriori(transactions_database, 0.3)
    print(lk)
    rules=generate_rules(lk, 0.8)
    print(rules)
    print_rules(rules)