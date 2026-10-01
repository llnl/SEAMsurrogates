# Neural Network Implementation TODOs

Remaining non-canonical patterns in `surmod/neural_network.py` to address later.

## 1. Switch from SGD to Adam Optimizer (PRIORITY: HIGH)

**Current (Line 168):**
```python
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
```

**Issue:** Plain SGD without momentum is outdated. Modern standard is Adam.

**Why Adam:**
- Adaptive learning rates per parameter
- Works well out-of-the-box with minimal tuning
- De facto standard for neural networks
- Default learning rate is typically 0.001 (vs 0.01 for SGD)

**Proposed Fix:**
```python
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
```

**Impact:** Will need to update default learning rate in scripts from 0.01 to 0.001 or make it optimizer-dependent.

---

## 2. DataLoader Shuffle Reproducibility (PRIORITY: MEDIUM)

**Current (Line 159):**
```python
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
```

**Issue:** `shuffle=True` uses non-reproducible randomness even with `torch.manual_seed()`.

**Proposed Fix:**
```python
generator = torch.Generator().manual_seed(seed)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
```

**Impact:** Ensures full reproducibility of training across runs.

---

## 3. Device Management (CPU/GPU) (PRIORITY: LOW)

**Issue:** Everything runs on CPU only. No way to use GPU acceleration.

**Canonical pattern:**
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
x_train = x_train.to(device)
x_test = x_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

# In training loop
for inputs, targets in train_loader:
    inputs, targets = inputs.to(device), targets.to(device)
    # ...
```

**Impact:** Enables GPU acceleration for larger datasets/models. Probably not critical for current educational use case.

---

## 4. Test vs Validation Naming (PRIORITY: LOW - Documentation)

**Current (Lines 197-202):**
```python
# Evaluate on the test set
model.eval()
with torch.no_grad():
    test_outputs = model(x_test)
    test_loss = criterion(test_outputs, y_test.view(-1, 1))
    test_losses.append(test_loss.item())
```

**Issue:** This is actually **validation** (used during training for monitoring), not testing. Evaluating on the true test set during training is data leakage.

**Options:**
1. Rename variables to `x_val`/`y_val`/`val_losses` throughout
2. Add documentation clarifying this is validation monitoring for educational purposes
3. Keep as-is if the pedagogical goal is simplicity over rigor

**Impact:** Mostly naming/documentation. Current behavior is fine for teaching, just misleading terminology.

---

## Summary

**Must fix:**
- [ ] Switch to Adam optimizer (#1)
- [ ] Add generator to DataLoader for reproducibility (#2)

**Nice to have:**
- [ ] Device management for GPU support (#3)
- [ ] Clarify test vs validation terminology (#4)

## Notes

- Already fixed: gradient accumulation, seed placement, plot_losses_verbose removal
- Current implementation prioritizes simplicity for educational use
- Consider making optimizer configurable rather than hardcoding Adam
