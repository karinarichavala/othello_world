import torch

print("=" * 50)
print("VERIFICACIÓN COMPLETA")
print("=" * 50)

# Información básica
print(f"PyTorch: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"Versión CUDA: {torch.version.cuda}")

if torch.cuda.is_available():
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"Memoria total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # CRÍTICO: Verificar que sm_120 esté soportado
    arch_list = torch.cuda.get_arch_list()
    print(f"\nArquitecturas soportadas:")
    print(arch_list)
    
    # Buscar sm_120
    has_sm120 = any('sm_120' in arch or 'sm120' in arch for arch in arch_list)
    print(f"\n✅ sm_120 soportado: {has_sm120}")
    
    # Test funcional
    print("\n" + "=" * 50)
    print("TEST FUNCIONAL")
    print("=" * 50)
    
    try:
        # Test simple
        x = torch.randn(1000, 1000, device='cuda')
        y = x @ x
        print("✅ Operación simple: OK")
        
        # Test más complejo (similar a SAE)
        batch = torch.randn(64, 512, device='cuda')
        linear = torch.nn.Linear(512, 256).cuda()
        output = linear(batch)
        print("✅ Red neuronal básica: OK")
        
        # Test con autograd (para entrenamiento)
        x = torch.randn(10, 10, device='cuda', requires_grad=True)
        y = (x ** 2).sum()
        y.backward()
        print("✅ Backpropagation: OK")
        
        print("\n🎉 TODO FUNCIONANDO PERFECTAMENTE!")
        print("Tu GPU está lista para entrenar SAEs con OthelloGPT")
        
    except Exception as e:
        print(f"❌ Error: {e}")
else:
    print("❌ CUDA no disponible")
