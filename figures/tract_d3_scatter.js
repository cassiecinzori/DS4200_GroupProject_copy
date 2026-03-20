const margin = { top: 28, right: 30, bottom: 68, left: 78 };
const outerWidth = 1080;
const outerHeight = 560;
const width = outerWidth - margin.left - margin.right;
const height = outerHeight - margin.top - margin.bottom;

const svg = d3.select('#chart')
    .attr('viewBox', [0, 0, outerWidth, outerHeight]);

const root = svg.append('g')
    .attr('transform', `translate(${margin.left},${margin.top})`);

const tooltip = d3.select('#tooltip');
const metricSelect = d3.select('#metric-select');
const metaLine = d3.select('#meta-line');

const formatPercent = d3.format('.1%');
const formatNumber = d3.format(',.0f');
const formatOneDecimal = d3.format(',.1f');
const formatIncome = d3.format('$,.0f');

const metricConfig = {
    complaint_change_per_1k: {
        label: 'Complaint change per 1k residents',
        accessor: d => d.complaintChangePer1k,
        domain: [-300, 300],
        format: d => formatOneDecimal(d),
    },
    complaints_per_1k_2025: {
        label: '2025 complaints per 1k residents',
        accessor: d => d.complaintsPer1k2025,
        domain: [0, 500],
        format: d => formatOneDecimal(d),
    },
    complaint_change_2025_minus_2015: {
        label: 'Raw complaint count change',
        accessor: d => d.complaintChangeRaw,
        domain: [-250, 250],
        format: d => formatNumber(d),
    },
};

const xScale = d3.scaleLinear().range([0, width]);
const yScale = d3.scaleLinear().range([height, 0]);
const radiusScale = d3.scaleSqrt().range([3.5, 13]);
const incomeScale = d3.scaleLinear()
    .range(['#c6dbef', '#6b9ecf', '#1f4e79'])
    .interpolate(d3.interpolateRgb);

const xGrid = root.append('g').attr('class', 'grid').attr('transform', `translate(0,${height})`);
const yGrid = root.append('g').attr('class', 'grid');
const xAxis = root.append('g').attr('class', 'axis').attr('transform', `translate(0,${height})`);
const yAxis = root.append('g').attr('class', 'axis');

const plot = root.append('g');

    root.append('text')
        .attr('x', width / 2)
        .attr('y', height + 52)
        .attr('text-anchor', 'middle')
        .attr('font-size', 14)
        .attr('font-weight', 600)
        .attr('fill', '#2c3e50')
        .text('Poverty rate');

const yLabel = root.append('text')
    .attr('transform', 'rotate(-90)')
    .attr('x', -height / 2)
    .attr('y', -54)
    .attr('text-anchor', 'middle')
    .attr('font-size', 14)
    .attr('font-weight', 600)
    .attr('fill', '#2c3e50');

let data = [];
let selectedMetric = metricSelect.property('value');
let pinnedId = null;

function preprocess(rows) {
    return rows.map(row => {
        const population = +row.population;
        const complaints2015 = +row.complaint_count_2015;
        const complaints2025 = +row.complaint_count_2025;
        const povertyRate = +row.poverty_rate / 100;
        const income = +row.median_household_income;
        const complaintsPer1k2015 = population > 0 ? (complaints2015 / population) * 1000 : 0;
        const complaintsPer1k2025 = population > 0 ? (complaints2025 / population) * 1000 : 0;

        return {
            ...row,
            population,
            povertyRate,
            income,
            complaintsPer1k2015,
            complaintsPer1k2025,
            complaintChangePer1k: complaintsPer1k2025 - complaintsPer1k2015,
            complaintChangeRaw: complaints2025 - complaints2015,
            label: row.NAME.replace('Census Tract ', 'Tract '),
        };
    }).filter(row => Number.isFinite(row.population) && Number.isFinite(row.income));
}

function renderMeta(config, valueAccessor) {
    const selected = pinnedId ? data.find(d => d.GEO_ID === pinnedId) : null;

    if (!selected) {
        metaLine.html(`
      <div><strong>Source:</strong> data/census_tract_ses_2023_with_311.csv</div>
      <div><strong>Observations:</strong> ${formatNumber(data.length)} tracts</div>
      <div><strong>Mean ${config.label}:</strong> ${config.format(d3.mean(data, valueAccessor))}</div>
      <div><strong>Click:</strong> pin a tract</div>
    `);
        return;
    }

    metaLine.html(`
    <div><strong>Selected tract:</strong> ${selected.label}</div>
    <div><strong>Poverty rate:</strong> ${formatPercent(selected.povertyRate)}</div>
    <div><strong>${config.label}:</strong> ${config.format(valueAccessor(selected))}</div>
    <div><strong>Median income:</strong> ${formatIncome(selected.income)}</div>
    <div><strong>Population:</strong> ${formatNumber(selected.population)}</div>
  `);
}

function render() {
    if (!data.length) return;

    const config = metricConfig[selectedMetric];
    const valueAccessor = config.accessor;
    const xDomain = d3.extent(data, d => d.povertyRate);
    const yDomain = d3.extent(data, valueAccessor);
    const yPad = ((yDomain[1] - yDomain[0]) * 0.12) || 1;

    xScale.domain([Math.max(0, xDomain[0] - 0.02), Math.min(1, xDomain[1] + 0.02)]);
    yScale.domain([
        Math.min(config.domain[0], yDomain[0] - yPad),
        Math.max(config.domain[1], yDomain[1] + yPad),
    ]).nice();
    radiusScale.domain(d3.extent(data, d => d.population));
    const incomeValues = data.map(d => d.income).filter(Number.isFinite).sort(d3.ascending);
    incomeScale.domain([
        d3.min(incomeValues),
        d3.quantileSorted(incomeValues, 0.5),
        d3.max(incomeValues),
    ]);

    xAxis.call(d3.axisBottom(xScale).tickFormat(formatPercent).ticks(8));
    yAxis.call(d3.axisLeft(yScale).ticks(8));
    xGrid.call(d3.axisBottom(xScale).tickSize(-height).tickFormat('').ticks(8));
    yGrid.call(d3.axisLeft(yScale).tickSize(-width).tickFormat('').ticks(8));
    yLabel.text(config.label);

    const points = plot.selectAll('circle').data(data, d => d.GEO_ID);

    points.join(
        enter => enter.append('circle')
            .attr('cx', d => xScale(d.povertyRate))
            .attr('cy', height)
            .attr('r', 0)
            .attr('fill', d => incomeScale(d.income))
            .attr('fill-opacity', 0.84)
            .attr('stroke', '#ffffff')
            .attr('stroke-width', 1.2)
            .on('mouseenter', function (event, d) {
                if (pinnedId && pinnedId !== d.GEO_ID) return;
                d3.select(this).attr('stroke', '#1a1a1a').attr('stroke-width', 2.5);
                tooltip.style('opacity', 1).html(`
          <strong>${d.label}</strong><br>
          Poverty rate: ${formatPercent(d.povertyRate)}<br>
          ${config.label}: ${config.format(valueAccessor(d))}<br>
          Median income: ${formatIncome(d.income)}<br>
          Population: ${formatNumber(d.population)}
        `);
            })
            .on('mousemove', function (event) {
                tooltip.style('left', `${event.pageX + 16}px`).style('top', `${event.pageY - 18}px`);
            })
            .on('mouseleave', function () {
                if (!pinnedId) {
                    d3.select(this).attr('stroke', '#ffffff').attr('stroke-width', 1.2);
                }
                tooltip.style('opacity', 0);
            })
            .on('click', function (event, d) {
                if (pinnedId === d.GEO_ID) {
                    pinnedId = null;
                    plot.selectAll('circle').classed('highlight', false).attr('opacity', 0.84);
                    renderMeta(config, valueAccessor);
                    tooltip.style('opacity', 0);
                    return;
                }

                pinnedId = d.GEO_ID;
                plot.selectAll('circle')
                    .classed('highlight', node => node.GEO_ID === d.GEO_ID)
                    .attr('opacity', node => node.GEO_ID === d.GEO_ID ? 1 : 0.18);

                tooltip.style('opacity', 1).html(`
          <strong>${d.label}</strong><br>
          Poverty rate: ${formatPercent(d.povertyRate)}<br>
          ${config.label}: ${config.format(valueAccessor(d))}<br>
          Median income: ${formatIncome(d.income)}<br>
          Population: ${formatNumber(d.population)}
        `);
                renderMeta(config, valueAccessor);
            })
            .call(enter => enter.transition().duration(700)
                .attr('cx', d => xScale(d.povertyRate))
                .attr('cy', d => yScale(valueAccessor(d)))
                .attr('r', d => radiusScale(d.population))),
        update => update.transition().duration(700)
            .attr('cx', d => xScale(d.povertyRate))
            .attr('cy', d => yScale(valueAccessor(d)))
            .attr('r', d => radiusScale(d.population))
            .attr('fill', d => incomeScale(d.income))
            .attr('opacity', d => pinnedId && d.GEO_ID !== pinnedId ? 0.18 : 0.84),
        exit => exit.transition().duration(200).attr('r', 0).remove()
    );

    renderMeta(config, valueAccessor);
}

metricSelect.on('change', function () {
    selectedMetric = this.value;
    if (pinnedId) {
        pinnedId = null;
        plot.selectAll('circle').classed('highlight', false).attr('opacity', 0.84);
    }
    render();
});

d3.csv('../data/census_tract_ses_2023_with_311.csv', d3.autoType)
    .then(rows => {
        data = preprocess(rows);
        render();
    })
    .catch(error => {
        d3.select('.page').append('p')
            .style('color', '#b91c1c')
            .style('font-weight', '700')
            .text(`Unable to load tract data: ${error.message}`);
        console.error(error);
    });
