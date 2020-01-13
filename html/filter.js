var filtersConfig = {
    base_path: 'tablefilter/',
    col_0: 'checklist',
    col_1: 'checklist',
    col_2: 'checklist',
    col_3: 'checklist',
    col_4: 'checklist',
    col_5: 'checklist',
    col_6: 'checklist',
    col_7: 'none',
    col_8: 'none',
    filters_row_index: 1,
    enable_checklist_reset_filter: false,
    alternate_rows: true,
    col_types: [
        'number', 'string', 'string',
        'string', 'string', 'string',
        'string', 'image', 'image'
    ],
    col_widths: [
        '80px', '150px', '130px',
        '110px', '170px', '150px',
        '150px', '210px', '210px'
    ]
};

var tf = new TableFilter('results', filtersConfig);
tf.init();

for (div of document.getElementsByClassName("div_checklist")) {
    div.style.height = 100;
}
