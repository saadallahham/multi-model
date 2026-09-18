import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import json
from sklearn.base import clone
from sklearn.metrics import accuracy_score, r2_score
from models import KernelNB, inverse_square, make_model, REG_IDS
from synthetic_data import generate, FEATURES, DEV_WELLS, APP_WELLS


class ModelTests(unittest.TestCase):
    def test_synthetic_reproducibility_and_physical_ranges(self):
        a, b = generate(120, 23), generate(120, 23)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(a.Well.nunique(), 5)
        self.assertTrue(a.Porosity.between(0,1).all())
        self.assertTrue((a.Permeability > 0).all())
        self.assertFalse(set(DEV_WELLS) & set(APP_WELLS))
        self.assertTrue(all(set(x.Facies) == set(range(4)) for _,x in a.groupby('Well')))

    def test_kernel_nb_probabilities(self):
        X = np.array([[-3.,-2.],[-2.,-3.],[2.,3.],[3.,2.]])
        y = np.array([4,4,9,9])
        model = clone(KernelNB()).fit(X,y)
        np.testing.assert_allclose(model.predict_proba(X).sum(axis=1),1)
        np.testing.assert_array_equal(model.predict(X),y)

    def test_inverse_square_duplicate_handling(self):
        weights = inverse_square(np.array([[0.,0.,2.],[1.,2.,4.]]))
        np.testing.assert_allclose(weights,[[1,1,0],[1,.25,.0625]])

    def test_full_registry_is_cloneable(self):
        for task in ['Facies','Porosity','Permeability']:
            for number in (range(1,26) if task == 'Facies' else REG_IDS):
                clone(make_model(number,task))
        self.assertEqual(len(REG_IDS),25)

    def test_preprocessing_uses_training_only(self):
        data = generate(120)
        train = data[data.Well == 'Well_1']
        model = make_model(4,'Facies').fit(train[FEATURES],train.Facies)
        scaler = model.named_steps['standardscaler']
        np.testing.assert_allclose(scaler.mean_,train[FEATURES].mean())
        held = data[data.Well == 'Well_4'][FEATURES].copy() + 100
        model.predict(held)
        np.testing.assert_allclose(scaler.mean_,train[FEATURES].mean())

    def test_log_permeability_inverse_transform(self):
        data = generate(120)
        model = make_model(26,'Permeability').fit(data[FEATURES],data.Permeability)
        pred = model.predict(data[FEATURES])
        self.assertTrue(np.isfinite(pred).all() and (pred > 0).all())

    def test_manuscript_regression_variants(self):
        data = generate(120)
        train = data[data.Well == 'Well_1']
        test = data[data.Well == 'Well_4']
        for number in [27, 28, 29, 30, 31]:
            model = make_model(number, 'Porosity').fit(train[FEATURES], train.Porosity)
            self.assertTrue(np.isfinite(model.predict(test[FEATURES])).all())

    @unittest.skipUnless(Path('outputs/metrics.csv').exists(), 'Run experiment first')
    def test_all_manuscript_figures_exist(self):
        folder = Path('outputs/manuscript_figures')
        manifest = json.loads((folder/'figure_manifest.json').read_text())
        self.assertEqual([e['number'] for e in manifest], list(range(1,20)))
        for number in range(1,20):
            for extension in ['png', 'pdf']:
                self.assertGreater((folder/f'figure_{number:02d}.{extension}').stat().st_size, 1000)

    @unittest.skipUnless(Path('outputs/metrics.csv').exists(), 'Run experiment first')
    def test_generated_metrics_match_predictions(self):
        predictions = pd.read_csv('outputs/predictions_all_models.csv')
        metrics = pd.read_csv('outputs/metrics.csv')
        self.assertEqual(len(metrics), (25 + 2*len(REG_IDS))*5)
        for (task,number,well), p in predictions.groupby(['Task','Model','Well']):
            m = metrics[(metrics.Task == task)&(metrics.Model == number)&(metrics.Well == well)].iloc[0]
            actual = accuracy_score(p.Truth,p.Prediction) if task == 'Facies' else r2_score(p.Truth,p.Prediction)
            self.assertAlmostEqual(actual, m.Accuracy if task == 'Facies' else m.R2, places=10)

    @unittest.skipUnless(Path('outputs/reapplied_predictions.csv').exists(), 'Run apply_models.py first')
    def test_saved_models_and_selection(self):
        original = pd.read_csv('outputs/application_predictions.csv').sort_values(['Well','Depth_m'])
        reloaded = pd.read_csv('outputs/reapplied_predictions.csv').sort_values(['Well','Depth_m'])
        columns = ['Facies_predicted','Porosity_predicted','Permeability_predicted']
        np.testing.assert_allclose(original[columns],reloaded[columns],rtol=1e-12)
        metrics = pd.read_csv('outputs/metrics.csv')
        selected = json.loads(Path('outputs/selected_models.json').read_text())
        for task, number in selected.items():
            dev = metrics[(metrics.Task == task)&(metrics.Split == 'development_cv')]
            score = 'MacroF1' if task == 'Facies' else 'R2'
            best = dev.groupby('Model')[score].mean().idxmax()
            self.assertEqual(number,best)


if __name__ == '__main__':
    unittest.main()
