def save (figs, d_paths, title):
    path = d_paths['charts'] / f'{title}.html'

    if type (figs) == list:
        with open(path, 'w') as file:
            for f in figs:
                file.write(f.to_html(full_html=False, include_plotlyjs='cdn'))

    else:
        figs.write_html(path)


def drop_categ_cols (df):
    categ_cols = df.select_dtypes(include=['category']).columns
    return df.drop(categ_cols,axis=1)


def make_formula (y_var, d_vars):
    formula = ''
    for cur_var, degree in d_vars.items():
        for cur_degree in range (1, degree+1):
            if cur_degree == 1:
                if formula == '':
                    formula = cur_var
                else:
                    formula += f" + {cur_var}"
            else: #degree > 1
                formula += f" + I({cur_var}**{degree})"
    return f"{y_var} ~ " + formula    