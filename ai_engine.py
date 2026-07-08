def analyze_image(image, environment, notes):

    if environment == "obra":
        risks = [
            "Risco de queda em altura",
            "Ausência ou uso incorreto de EPI",
            "Materiais espalhados causando risco de acidente"
        ]
        level = "high"

    elif environment == "industria":
        risks = [
            "Risco relacionado a máquinas e equipamentos",
            "Necessidade de inspeção do ambiente",
            "Verificar utilização correta de EPIs"
        ]
        level = "medium"

    else:
        risks = [
            "Ambiente necessita de avaliação de segurança",
            "Possível falta de organização no local"
        ]
        level = "medium"


    return {
        "analysis": {
            "environment_analyzed": environment,
            "risk_detected": True,
            "risk_level": level,
            "risks": risks,

            "recommendations": [
                "Verificar equipamentos de proteção individual.",
                "Manter o ambiente organizado.",
                "Realizar inspeções periódicas."
            ]
        },

        "image_received": True,
        "image_size_bytes": len(image),
        "notes_received": notes
    }