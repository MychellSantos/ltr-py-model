categorias_core = {
    145: "Eletro",
    143: "Eletro",
    3267: "Smartphones",
    1588: "Televisores"
}

def is_core_categoria(categoria_id):
    return categoria_id in categorias_core

def calcula_grade(core, _source):
    mais_vendidos = _source.get('maisVendidos', 0) or 0
    popularidade = _source.get('popularidade', 0) or 0
    classificacaoMedia = _source.get('classificacaoMedia', 0) or 0
    pontos = {
        'core': 2 if core else 0,
        'mais_vendidos': 1 if mais_vendidos > 50 else 0,
        'popularidade': 1 if popularidade > 200 else 0,
        'classificacaoMedia': 2 if classificacaoMedia > 3 else 0,
    }
    return min(sum(pontos.values()), 4)