# Processor
Processor module has a simple API. You need to make three steps to snandardize and convert data into output format:
1. Create Processing object with specifying scaling type (z-scaling applies by default)

```python
import processor as pr
proc=pr.Processing()
```
    
2. Fit processing object with specifying file name of train dataset:

```python
proc.fit('train_test2.tsv')
```

3. Convert specified test dataset into output format:

```python
proc.transform('train_test2.tsv')
```

'test_proc.tsv' file will be generated.

Note that here there is no convertion for other feature types except 2. The problem is that I can't understand what exactly means on other feature types:(
