/*+ HashJoin(ct t k mk mi at it1 mc cn)
 HashJoin(ct t k mk mi at it1 mc)
 HashJoin(t k mk mi at it1 mc)
 MergeJoin(t k mk mi at it1)
 HashJoin(t k mk mi at)
 HashJoin(t k mk mi)
 HashJoin(t k mk)
 HashJoin(k mk)
 SeqScan(ct)
 SeqScan(t)
 IndexScan(k)
 IndexScan(mk)
 SeqScan(mi)
 IndexScan(at)
 SeqScan(it1)
 SeqScan(mc)
 IndexScan(cn)
 Leading(((ct (((((t (k mk)) mi) at) it1) mc)) cn)) */
SELECT MIN(mi.info) AS release_date,
       MIN(t.title) AS modern_american_internet_movie
FROM aka_title AS at,
     company_name AS cn,
     company_type AS ct,
     info_type AS it1,
     keyword AS k,
     movie_companies AS mc,
     movie_info AS mi,
     movie_keyword AS mk,
     title AS t
WHERE cn.country_code = '[us]'
  AND it1.info = 'release dates'
  AND mi.note LIKE '%internet%'
  AND mi.info IS NOT NULL
  AND (mi.info LIKE 'USA:% 199%'
       OR mi.info LIKE 'USA:% 200%')
  AND t.production_year > 1990
  AND t.id = at.movie_id
  AND t.id = mi.movie_id
  AND t.id = mk.movie_id
  AND t.id = mc.movie_id
  AND mk.movie_id = mi.movie_id
  AND mk.movie_id = mc.movie_id
  AND mk.movie_id = at.movie_id
  AND mi.movie_id = mc.movie_id
  AND mi.movie_id = at.movie_id
  AND mc.movie_id = at.movie_id
  AND k.id = mk.keyword_id
  AND it1.id = mi.info_type_id
  AND cn.id = mc.company_id
  AND ct.id = mc.company_type_id;

