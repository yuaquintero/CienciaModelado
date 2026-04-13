import torch
import torch.nn as nn

class BuckiNetLayer(nn.Module):
    """
    Capa base de BuckiNet CORREGIDA.
    Maneja valores negativos usando valores absolutos.
    """
    def __init__(self, num_inputs, num_pi_groups):
        super(BuckiNetLayer, self).__init__()
        # Inicialización más pequeña para evitar explosiones
        self.Phi_p = nn.Parameter(torch.randn(num_inputs, num_pi_groups) * 0.1)

    def forward(self, x):
        # SOLUCIÓN: Tomar valores absolutos para evitar log(negativo)
        # Esto es físicamente válido porque las variables físicas 
        # (longitud, tiempo, velocidad, gravedad) son siempre positivas
        x_abs = torch.abs(x) + 1e-8  # Pequeño epsilon para evitar log(0)
        
        # Logaritmo de valores positivos
        log_x = torch.log(x_abs)
        
        # Multiplicación en espacio logarítmico
        pi_groups_log = torch.matmul(log_x, self.Phi_p)
        
        # Clamp para evitar exp() explosivo
        pi_groups_log = torch.clamp(pi_groups_log, min=-50, max=50)
        
        # Volver al espacio lineal
        pi_groups = torch.exp(pi_groups_log)
        
        return pi_groups


class FullBuckiNet(nn.Module):
    
    def __init__(self, num_inputs, num_pi_groups, num_outputs, hidden_neurons=32):
        super(FullBuckiNet, self).__init__()
        
        self.pi_layer = BuckiNetLayer(num_inputs, num_pi_groups)
        
        self.mlp = nn.Sequential(
            nn.Linear(num_pi_groups, hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, num_outputs)
        )

    def forward(self, x):
        pi_groups = self.pi_layer(x)
        
        # NUEVO: Clamp para evitar valores extremos
        pi_groups = torch.clamp(pi_groups, min=1e-6, max=1e6)
        
        # NUEVO: Log transform para comprimir el rango
        pi_groups_log = torch.log(pi_groups)
        
        output = self.mlp(pi_groups_log)
        
        return output, self.pi_layer.Phi_p

def buckinet_loss(predicciones, Y_train, Phi_p, D_p, lambda_null=1.0, l1_reg=1e-4, l2_reg=1e-5):
    """
    Función de pérdida informada por física con parámetros más conservadores.
    """
    # 1. Error de reconstrucción
    mse_loss = nn.MSELoss()(predicciones, Y_train)
    
    # 2. Penalización del Espacio Nulo (Física)
    # Verificar dimensiones de D_p y Phi_p
    # D_p debería ser (dimensiones, num_inputs) o (num_inputs, dimensiones)
    # Ajusta según tu implementación
    if D_p.shape[1] == Phi_p.shape[0]:  # D_p (dims, inputs), Phi_p (inputs, groups)
        null_space_penalty = torch.sum(torch.square(torch.matmul(D_p, Phi_p)))
    else:  # D_p (inputs, dims), Phi_p (inputs, groups)
        # Necesitamos D_p.T @ Phi_p? Depende de la forma
        null_space_penalty = torch.sum(torch.square(torch.matmul(Phi_p.T, D_p)))
    
    # 3. Regularizaciones
    l1_penalty = torch.sum(torch.abs(Phi_p))
    l2_penalty = torch.sum(torch.square(Phi_p))
    
    # Suma ponderada
    total_loss = mse_loss + (lambda_null * null_space_penalty) + (l1_reg * l1_penalty) + (l2_reg * l2_penalty)
    
    return total_loss